using System;
using System.IO;
using UnityEngine;
using UnityEngine.InputSystem;

public class WorldModelPlannerAgent : MonoBehaviour
{
    public GameManager Game;
    public BrainHUDController BrainHUD;
    public RogueTransitionRecorder TransitionRecorder;
    public TextAsset ModelWeightsJson;
    public TextAsset MetricsJson;
    public string StreamingAssetsModelPath = "world_model_weights.json";
    public string StreamingAssetsMetricsPath = "world_model_metrics.json";
    public bool StartNewGameOnStart = true;
    public bool StartWithModelEnabled = false;
    public bool DisableHumanInput = false;
    public bool InstantActions = false;
    public bool RecordPlannerTransitions = true;
    public float DecisionIntervalSeconds = 0.45f;
    [Range(1, 5)]
    public int PlanningHorizon = 3;
    [Range(0f, 1f)]
    public float RolloutDiscount = 0.85f;
    public float HeuristicBlend = 0.05f;
    public float StepPenalty = 0.001f;
    public float InvalidActionPenalty = 0.025f;
    public float ProgressReward = 0.05f;
    public float FoodDeltaReward = 0.01f;
    public float ExitReward = 1.0f;
    public float GameOverPenalty = -1.0f;
    public Key ToggleKey = Key.M;

    [Header("LeWorldModel sidecar (M5)")]
    [Tooltip("When true, the agent queries the Python sidecar (POST /plan_actions) as the primary scorer and falls back to the local mission planner only when the sidecar is unreachable.")]
    public bool UseSidecar = false;
    [Tooltip("Pre-existing LewmClient component. If left null, one will be added to this GameObject automatically when UseSidecar is true.")]
    public LewmClient SidecarClient;
    public string SidecarBaseUrl = "http://127.0.0.1:5555";
    [Range(4, 256)]
    public int SidecarCandidates = 64;
    [Range(1, 8)]
    public int SidecarTopK = 3;
    [Tooltip("0 = let the sidecar pick a fresh seed each request (stochastic).")]
    public int SidecarSeed = 0;
    [Tooltip("How long to wait between /info probes when the sidecar appears offline. Successful probes also refresh the cached max_horizon.")]
    [Range(0.5f, 60f)]
    public float SidecarHealthProbeSeconds = 5f;
    public float SidecarRequestTimeoutSeconds = 1.0f;

    private WorldModelWeights m_Model;
    private BrainMetrics m_Metrics = BrainMetrics.Empty();
    private float m_NextDecisionTime;
    private bool m_IsModelControlEnabled;
    private bool m_UiBound;
    private PendingTransition m_PendingTransition;

    // ---- sidecar state (M5) ----
    private bool m_SidecarClientReady;
    private bool m_SidecarOnline;
    private bool m_SidecarInfoLoaded;
    private bool m_SidecarRequestInFlight;
    private float m_NextHealthProbe;
    private int m_SidecarMaxHorizon = 1;
    private int m_SidecarActionDim = RogueObservationBuilder.ActionCount;

    private void Start()
    {
        EnsureReferences();
        LoadModel();
        LoadMetrics();

        if (Game == null)
        {
            Debug.LogWarning("WorldModelPlannerAgent could not find a GameManager.");
            return;
        }

        Game.EnsureInitialized();
        BindUi();

        if (StartNewGameOnStart)
        {
            Game.StartNewGame();
            TransitionRecorder?.BeginEpisode();
        }

        SetModelControl(StartWithModelEnabled || DisableHumanInput);

        if (UseSidecar)
        {
            EnsureSidecarClient();
            m_NextHealthProbe = Time.time;
        }

        PublishBrainFrame();
    }

    private void Update()
    {
        EnsureReferences();
        BindUi();
        HandleToggleInput();

        if (Game != null && Game.PlayerController != null && !Game.PlayerController.IsMoving)
        {
            FlushPendingTransition();
        }

        if (UseSidecar)
        {
            EnsureSidecarClient();
            MaybeProbeSidecar();
        }

        if (!m_IsModelControlEnabled)
        {
            return;
        }

        if (Game == null || Game.IsGameOver || Game.PlayerController.IsMoving)
        {
            return;
        }

        if (Time.time < m_NextDecisionTime)
        {
            return;
        }

        // Sidecar path: fire one async /plan_actions request and apply
        // the first action of the returned best plan in the callback.
        // The decision-interval bump happens inside the callback so we
        // do NOT race-fire while a request is in flight.
        if (UseSidecar
            && m_SidecarClientReady
            && m_SidecarOnline
            && m_SidecarInfoLoaded
            && !m_SidecarRequestInFlight)
        {
            TryStepViaSidecar();
            return;
        }

        // Fallback path (mission-heuristic BrainPlanner, unchanged
        // behavior). This runs when the user has not enabled sidecar
        // control or the sidecar is unreachable / mid-recovery.
        m_NextDecisionTime = Time.time + DecisionIntervalSeconds;
        DecideAndStepViaBrainPlanner();
    }

    public void ToggleModelControl()
    {
        SetModelControl(!m_IsModelControlEnabled);
    }

    public void SetModelControl(bool enabled)
    {
        m_IsModelControlEnabled = enabled;

        if (Game != null && Game.PlayerController != null)
        {
            if (m_IsModelControlEnabled && Game.IsGameOver)
            {
                Game.StartNewGame();
                TransitionRecorder?.BeginEpisode();
            }

            bool humanControlEnabled = !m_IsModelControlEnabled;
            Game.PlayerController.EnableHumanInput = humanControlEnabled;

            if (Game.IsGameOver)
            {
                Game.SetPlayerInputEnabled(false);
                if (humanControlEnabled)
                {
                    Game.PlayerController.StartNewGameAction.Enable();
                }
            }
            else
            {
                Game.SetPlayerInputEnabled(humanControlEnabled);
            }
        }

        m_NextDecisionTime = Time.time;
        UpdateUi();
        PublishBrainFrame();
    }

    public int ChooseAction()
    {
        return SelectedAction(BuildBrainFrame());
    }

    private BrainHUDData BuildBrainFrame()
    {
        EnsureReferences();
        return BrainPlanner.Decide(
            Game,
            m_Model,
            m_Metrics,
            PlanningHorizon,
            RolloutDiscount,
            HeuristicBlend,
            m_IsModelControlEnabled);
    }

    private void PublishBrainFrame()
    {
        if (BrainHUD != null && Game != null && !Game.IsGameOver)
        {
            BrainHUD.UpdateBrain(BuildBrainFrame());
        }
    }

    private static int SelectedAction(BrainHUDData brainFrame)
    {
        if (brainFrame == null || brainFrame.actionRanking == null || brainFrame.actionRanking.Length == 0)
        {
            return 0;
        }

        for (int i = 0; i < brainFrame.actionRanking.Length; i++)
        {
            if (brainFrame.actionRanking[i].selected)
            {
                return brainFrame.actionRanking[i].action;
            }
        }

        return brainFrame.actionRanking[0].action;
    }

    private void HandleToggleInput()
    {
        if (ToggleKey != Key.None && Keyboard.current != null && Keyboard.current[ToggleKey].wasPressedThisFrame)
        {
            ToggleModelControl();
        }
    }

    private void LoadModel()
    {
        try
        {
            string json = null;
            if (ModelWeightsJson != null)
            {
                json = ModelWeightsJson.text;
            }
            else
            {
                string path = Path.Combine(Application.streamingAssetsPath, StreamingAssetsModelPath);
                if (File.Exists(path))
                {
                    json = File.ReadAllText(path);
                }
            }

            if (!string.IsNullOrWhiteSpace(json))
            {
                m_Model = WorldModelWeights.FromJson(json);
                Debug.Log("Loaded world model planner weights.");
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"Could not load world model weights; planner will use map heuristic fallback. {ex.Message}");
            m_Model = null;
        }
    }

    private void LoadMetrics()
    {
        try
        {
            string json = null;
            if (MetricsJson != null)
            {
                json = MetricsJson.text;
            }
            else
            {
                string path = Path.Combine(Application.streamingAssetsPath, StreamingAssetsMetricsPath);
                if (File.Exists(path))
                {
                    json = File.ReadAllText(path);
                }
            }

            m_Metrics = !string.IsNullOrWhiteSpace(json)
                ? JsonUtility.FromJson<BrainMetrics>(json)
                : BrainMetrics.Empty();
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"Could not load world model metrics. {ex.Message}");
            m_Metrics = BrainMetrics.Empty();
        }
    }

    private void BindUi()
    {
        if (m_UiBound || Game == null)
        {
            return;
        }

        Game.EnsureInitialized();
        if (Game.UIManager == null)
        {
            return;
        }

        Game.UIManager.RegisterWorldModelToggle(ToggleModelControl);
        m_UiBound = true;
        UpdateUi();
    }

    private void UpdateUi()
    {
        if (Game != null && Game.UIManager != null)
        {
            Game.UIManager.SetWorldModelStatus(m_IsModelControlEnabled, m_Model != null && m_Model.IsLoaded);
        }
    }

    private void EnsureReferences()
    {
        if (Game == null)
        {
            Game = GameManager.Instance != null
                ? GameManager.Instance
                : FindAnyObjectByType<GameManager>();
        }

        if (BrainHUD == null)
        {
            BrainHUD = GetComponent<BrainHUDController>();
            if (BrainHUD == null)
            {
                BrainHUD = FindAnyObjectByType<BrainHUDController>();
            }
        }

        if (TransitionRecorder == null)
        {
            TransitionRecorder = GetComponent<RogueTransitionRecorder>();
            if (TransitionRecorder == null)
            {
                TransitionRecorder = FindAnyObjectByType<RogueTransitionRecorder>();
            }

            if (TransitionRecorder == null && RecordPlannerTransitions)
            {
                TransitionRecorder = gameObject.AddComponent<RogueTransitionRecorder>();
            }
        }
    }

    // ------------------------------------------------------------------
    // Sidecar wiring (M5)
    // ------------------------------------------------------------------

    private void EnsureSidecarClient()
    {
        if (!UseSidecar)
        {
            return;
        }

        if (m_SidecarClientReady)
        {
            if (SidecarClient != null)
            {
                SidecarClient.BaseUrl = SidecarBaseUrl;
                SidecarClient.TimeoutSeconds = SidecarRequestTimeoutSeconds;
            }
            return;
        }

        if (SidecarClient == null)
        {
            SidecarClient = GetComponent<LewmClient>();
            if (SidecarClient == null)
            {
                SidecarClient = gameObject.AddComponent<LewmClient>();
            }
        }

        SidecarClient.BaseUrl = SidecarBaseUrl;
        SidecarClient.TimeoutSeconds = SidecarRequestTimeoutSeconds;
        m_SidecarClientReady = true;
    }

    private void MaybeProbeSidecar()
    {
        if (!m_SidecarClientReady || m_SidecarRequestInFlight)
        {
            return;
        }

        if (Time.time < m_NextHealthProbe)
        {
            return;
        }

        m_NextHealthProbe = Time.time + Mathf.Max(0.5f, SidecarHealthProbeSeconds);
        StartCoroutine(SidecarClient.Info(OnSidecarInfo, OnSidecarProbeFailed));
    }

    private void OnSidecarInfo(LewmClient.InfoResponse info)
    {
        if (info == null)
        {
            m_SidecarOnline = false;
            m_SidecarInfoLoaded = false;
            return;
        }

        m_SidecarOnline = true;
        m_SidecarInfoLoaded = true;
        m_SidecarMaxHorizon = Mathf.Max(1, info.max_horizon);
        m_SidecarActionDim = Mathf.Max(1, info.action_dim);

        if (m_SidecarActionDim != RogueObservationBuilder.ActionCount)
        {
            Debug.LogWarning(
                $"Sidecar reports action_dim={m_SidecarActionDim} but Unity ActionCount={RogueObservationBuilder.ActionCount}. " +
                "Plans may not map cleanly; falling back to mission planner if requests fail.");
        }
    }

    private void OnSidecarProbeFailed(string err)
    {
        if (m_SidecarOnline)
        {
            // Only log the transition from online -> offline; further
            // failures stay silent so the console does not spam every
            // probe interval while the sidecar is down.
            Debug.LogWarning($"Sidecar /info probe failed: {err}. Falling back to mission planner.");
        }
        m_SidecarOnline = false;
        m_SidecarInfoLoaded = false;
    }

    private void TryStepViaSidecar()
    {
        m_SidecarRequestInFlight = true;

        // Snapshot pre-step state. The game state cannot mutate between
        // here and the callback because we already gated on
        // ``Game.PlayerController.IsMoving == false`` and the agent
        // owns input (human input disabled while m_IsModelControlEnabled).
        var snapshot = SnapshotPreStep();

        int requestHorizon = Mathf.Clamp(PlanningHorizon, 1, Mathf.Max(1, m_SidecarMaxHorizon));
        var request = new LewmClient.PlanRequest
        {
            board_width = snapshot.previousBoardWidth,
            board_height = snapshot.previousBoardHeight,
            board_state = snapshot.previousBoardState,
            horizon = requestHorizon,
            num_candidates = Mathf.Max(1, SidecarCandidates),
            top_k = Mathf.Clamp(SidecarTopK, 1, 8),
            discount = RolloutDiscount,
            done_penalty = 1.0f,
            seed = SidecarSeed,
        };

        StartCoroutine(SidecarClient.PlanActions(
            request,
            plan => OnSidecarPlanReceived(plan, snapshot),
            err => OnSidecarPlanFailed(err, snapshot)));
    }

    private PendingTransition SnapshotPreStep()
    {
        return new PendingTransition
        {
            observation = RogueObservationBuilder.Build(Game),
            previousLevel = Game.CurrentLevel,
            previousFood = Game.CurrentFoodAmount,
            previousDistance = Game.DistanceToExit(Game.PlayerCellPosition),
            previousBoardWidth = PixelObservationBuilder.GetBoardWidth(Game),
            previousBoardHeight = PixelObservationBuilder.GetBoardHeight(Game),
            previousBoardState = PixelObservationBuilder.BuildCellCodes(Game),
        };
    }

    private void OnSidecarPlanReceived(LewmClient.PlanResponse response, PendingTransition snapshot)
    {
        m_SidecarRequestInFlight = false;
        m_NextDecisionTime = Time.time + DecisionIntervalSeconds;

        // Stale-check: user may have toggled off, the level may have
        // restarted, or the player may have started moving via some
        // other code path while the HTTP round-trip was in flight.
        if (!m_IsModelControlEnabled
            || Game == null
            || Game.IsGameOver
            || Game.PlayerController.IsMoving)
        {
            return;
        }

        var summary = BrainPlanner.SidecarPlanSummary.FromPlanResponse(response);
        if (summary == null)
        {
            Debug.LogWarning("Sidecar /plan_actions returned malformed plan; falling back to mission planner.");
            m_SidecarOnline = false;
            FallbackStepAfterSidecarFailure(snapshot);
            return;
        }

        var brainFrame = BrainPlanner.DecideWithSidecarPlan(
            Game,
            summary,
            m_Metrics,
            m_IsModelControlEnabled);
        BrainHUD?.UpdateBrain(brainFrame);

        StepWithAction(SelectedAction(brainFrame), snapshot);
    }

    private void OnSidecarPlanFailed(string err, PendingTransition snapshot)
    {
        m_SidecarRequestInFlight = false;
        m_SidecarOnline = false;
        m_NextHealthProbe = Time.time + 0.5f;  // re-probe sooner after a failure
        m_NextDecisionTime = Time.time + DecisionIntervalSeconds;
        Debug.LogWarning($"Sidecar /plan_actions failed: {err}. Falling back to mission planner for this step.");
        FallbackStepAfterSidecarFailure(snapshot);
    }

    private void FallbackStepAfterSidecarFailure(PendingTransition snapshot)
    {
        if (!m_IsModelControlEnabled
            || Game == null
            || Game.IsGameOver
            || Game.PlayerController.IsMoving)
        {
            return;
        }

        var brainFrame = BuildBrainFrame();
        BrainHUD?.UpdateBrain(brainFrame);
        StepWithAction(SelectedAction(brainFrame), snapshot);
    }

    private void StepWithAction(int action, PendingTransition snapshot)
    {
        bool accepted = Game.PlayerController.TryStep(
            RogueObservationBuilder.ActionToDirection(action),
            smoothMovement: !InstantActions);

        if (RecordPlannerTransitions && TransitionRecorder != null)
        {
            snapshot.action = action;
            snapshot.accepted = accepted;
            m_PendingTransition = snapshot;
            if (!accepted || InstantActions)
            {
                FlushPendingTransition();
            }
        }
    }

    // ------------------------------------------------------------------
    // BrainPlanner (mission heuristic) path
    // ------------------------------------------------------------------

    private void DecideAndStepViaBrainPlanner()
    {
        var brainFrame = BuildBrainFrame();
        BrainHUD?.UpdateBrain(brainFrame);
        int action = SelectedAction(brainFrame);
        var snapshot = SnapshotPreStep();
        StepWithAction(action, snapshot);
    }

    private void FlushPendingTransition()
    {
        if (m_PendingTransition == null || TransitionRecorder == null || Game == null)
        {
            return;
        }

        bool reachedExit = Game.CurrentLevel > m_PendingTransition.previousLevel;
        int currentDistance = reachedExit ? 0 : Game.DistanceToExit(Game.PlayerCellPosition);
        float reward = -StepPenalty;

        if (!m_PendingTransition.accepted)
        {
            reward -= InvalidActionPenalty;
        }

        reward += (m_PendingTransition.previousDistance - currentDistance) * ProgressReward;
        reward += (Game.CurrentFoodAmount - m_PendingTransition.previousFood) * FoodDeltaReward;

        bool done = reachedExit || Game.IsGameOver;
        string outcome = "running";
        if (reachedExit)
        {
            reward += ExitReward;
            outcome = "exit";
        }
        else if (Game.IsGameOver)
        {
            reward += GameOverPenalty;
            outcome = "game_over";
        }

        TransitionRecorder.Record(
            m_PendingTransition.observation,
            m_PendingTransition.action,
            reward,
            RogueObservationBuilder.Build(Game),
            done,
            Game.CurrentLevel,
            Game.CurrentFoodAmount,
            outcome,
            m_PendingTransition.previousBoardWidth,
            m_PendingTransition.previousBoardHeight,
            m_PendingTransition.previousBoardState,
            PixelObservationBuilder.BuildCellCodes(Game));
        m_PendingTransition = null;
    }

    private class PendingTransition
    {
        public float[] observation;
        public int action;
        public bool accepted;
        public int previousLevel;
        public int previousFood;
        public int previousDistance;
        public int previousBoardWidth;
        public int previousBoardHeight;
        public int[] previousBoardState;
    }
}
