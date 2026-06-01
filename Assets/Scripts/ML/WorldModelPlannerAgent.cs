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
    private AgentMode m_AgentMode = AgentMode.HumanOnly;
    private CoachSessionLogger m_CoachLogger;
    private float m_NextDecisionTime;
    private bool m_UiBound;
    private bool m_GameEventsBound;
    private bool m_PlayerEventsBound;
    private bool m_RunInitialized;
    private bool m_CoachLogRunActive;
    private PendingTransition m_PendingTransition;
    private GameManager.RunEndedEvent m_PendingRunEndedEvent;
    private PlayerController m_BoundPlayerController;

    private BrainHUDData m_CachedCoachFrame;
    private float[] m_CachedCoachScores;
    private Vector2Int m_CachedCoachCell;
    private int m_CachedCoachLevel;
    private int m_CachedCoachFood;

    private int m_SidecarRequestVersion;

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
        BindGameEvents();
        BindPlayerEvents();
        BindUi();

        if (UseSidecar)
        {
            EnsureSidecarClient();
            m_NextHealthProbe = Time.time;
        }

        SetAgentMode(StartWithModelEnabled || DisableHumanInput
            ? AgentMode.AIAutonomous
            : AgentMode.HumanOnly);

        if (StartNewGameOnStart && Game.CurrentLevel == 0 && !Game.IsGameOver)
        {
            Game.StartNewGame();
        }
        else if (!m_RunInitialized && Game.CurrentLevel > 0 && !Game.IsGameOver)
        {
            HandleRunStarted();
        }

        PublishForCurrentMode();
    }

    private void Update()
    {
        EnsureReferences();
        BindGameEvents();
        BindPlayerEvents();
        BindUi();
        HandleToggleInput();

        if (Game != null && Game.PlayerController != null && !Game.PlayerController.IsMoving)
        {
            FlushPendingTransition();
            FinalizePendingRunEndedIfReady();
        }

        if (UseSidecar)
        {
            EnsureSidecarClient();
            MaybeProbeSidecar();
        }

        switch (m_AgentMode)
        {
            case AgentMode.CoachMode:
                UpdateCoachMode();
                break;
            case AgentMode.AIAutonomous:
                UpdateAutonomousMode();
                break;
            default:
                break;
        }
    }

    private void OnDestroy()
    {
        UnbindGameEvents();
        UnbindPlayerEvents();
    }

    public void ToggleModelControl()
    {
        SetAgentMode(NextMode(m_AgentMode));
    }

    public void SetModelControl(bool enabled)
    {
        SetAgentMode(enabled ? AgentMode.AIAutonomous : AgentMode.HumanOnly);
    }

    public int ChooseAction()
    {
        return SelectedAction(BuildLocalBrainFrame(m_AgentMode));
    }

    private void SetAgentMode(AgentMode mode)
    {
        if (m_AgentMode == mode)
        {
            Game?.SetDemoRunMode(mode);
            UpdateUi();
            return;
        }

        m_AgentMode = mode;
        Game?.SetDemoRunMode(mode);
        InvalidateSidecarRequests();

        if (Game != null && Game.PlayerController != null)
        {
            bool humanControlEnabled = mode != AgentMode.AIAutonomous;
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
        if (mode == AgentMode.CoachMode)
        {
            EnsureCoachLoggingSession();
        }

        UpdateUi();
        PublishForCurrentMode();
    }

    private void UpdateCoachMode()
    {
        if (Game == null || Game.IsGameOver || Game.PlayerController == null || Game.PlayerController.IsMoving)
        {
            return;
        }

        if (Time.time < m_NextDecisionTime)
        {
            return;
        }

        if (UseSidecar
            && m_SidecarClientReady
            && m_SidecarOnline
            && m_SidecarInfoLoaded
            && !m_SidecarRequestInFlight)
        {
            RequestCoachSidecarSuggestion();
            return;
        }

        m_NextDecisionTime = Time.time + DecisionIntervalSeconds;
        UpdateCoachFrame(BuildLocalBrainFrame(AgentMode.CoachMode));
    }

    private void UpdateAutonomousMode()
    {
        if (Game == null || Game.IsGameOver || Game.PlayerController == null || Game.PlayerController.IsMoving)
        {
            return;
        }

        if (Time.time < m_NextDecisionTime)
        {
            return;
        }

        if (UseSidecar
            && m_SidecarClientReady
            && m_SidecarOnline
            && m_SidecarInfoLoaded
            && !m_SidecarRequestInFlight)
        {
            RequestAutonomousSidecarStep();
            return;
        }

        m_NextDecisionTime = Time.time + DecisionIntervalSeconds;
        DecideAndStepViaBrainPlanner();
    }

    private void PublishForCurrentMode()
    {
        if (BrainHUD == null)
        {
            return;
        }

        if (m_AgentMode == AgentMode.HumanOnly)
        {
            BrainHUD.UpdateBrain(BuildHumanOnlyFrame());
            return;
        }

        if (m_AgentMode == AgentMode.CoachMode)
        {
            UpdateCoachFrame(GetCoachFrameForCurrentState());
            return;
        }

        BrainHUD.UpdateBrain(BuildLocalBrainFrame(AgentMode.AIAutonomous));
    }

    private BrainHUDData BuildHumanOnlyFrame()
    {
        return new BrainHUDData
        {
            aiEnabled = false,
            agentMode = AgentMode.HumanOnly,
            isCoachMode = false,
            modelLoaded = m_Model != null && m_Model.IsLoaded,
            modeLabel = "PLAYER CONTROL",
            selectedAction = "NONE",
            suggestedActionIndex = -1,
            suggestedActionName = "none",
            explanation = "Human-only mode active.",
            riskWarning = string.Empty,
            riskLevel = "low",
            metrics = m_Metrics ?? BrainMetrics.Empty(),
        };
    }

    private BrainHUDData BuildLocalBrainFrame(AgentMode agentMode)
    {
        return DecorateBrainFrame(BrainPlanner.Decide(
            Game,
            m_Model,
            m_Metrics,
            PlanningHorizon,
            RolloutDiscount,
            HeuristicBlend,
            agentMode));
    }

    private BrainHUDData BuildSidecarBrainFrame(BrainPlanner.SidecarPlanSummary summary, AgentMode agentMode)
    {
        return DecorateBrainFrame(BrainPlanner.DecideWithSidecarPlan(
            Game,
            summary,
            m_Metrics,
            agentMode));
    }

    private BrainHUDData DecorateBrainFrame(BrainHUDData frame)
    {
        if (frame == null)
        {
            return BuildHumanOnlyFrame();
        }

        frame.agentMode = m_AgentMode;
        frame.aiEnabled = m_AgentMode != AgentMode.HumanOnly;
        frame.isCoachMode = m_AgentMode == AgentMode.CoachMode;
        frame.suggestedActionIndex = SelectedAction(frame);
        frame.suggestedActionName = frame.suggestedActionIndex >= 0
            ? RogueObservationBuilder.ActionName(frame.suggestedActionIndex)
            : "none";
        frame.complianceRate = m_CoachLogger != null ? m_CoachLogger.complianceRate : 0f;
        frame.sessionFollowedSteps = m_CoachLogger != null ? m_CoachLogger.followedSteps : 0;
        frame.sessionTotalSteps = m_CoachLogger != null ? m_CoachLogger.totalSteps : 0;
        frame.riskWarning = string.IsNullOrWhiteSpace(frame.riskWarning)
            ? "Risk: waiting for suggestion."
            : frame.riskWarning;
        frame.riskLevel = string.IsNullOrWhiteSpace(frame.riskLevel)
            ? "blocked"
            : frame.riskLevel;
        return frame;
    }

    private BrainHUDData GetCoachFrameForCurrentState()
    {
        if (IsCachedCoachFrameCurrent() && m_CachedCoachFrame != null)
        {
            return m_CachedCoachFrame;
        }

        return BuildLocalBrainFrame(AgentMode.CoachMode);
    }

    private bool IsCachedCoachFrameCurrent()
    {
        return m_CachedCoachFrame != null
            && Game != null
            && Game.PlayerCellPosition == m_CachedCoachCell
            && Game.CurrentLevel == m_CachedCoachLevel
            && Game.CurrentFoodAmount == m_CachedCoachFood;
    }

    private void UpdateCoachFrame(BrainHUDData frame)
    {
        if (frame == null || Game == null)
        {
            return;
        }

        m_CachedCoachFrame = frame;
        m_CachedCoachScores = ExtractActionScores(frame);
        m_CachedCoachCell = Game.PlayerCellPosition;
        m_CachedCoachLevel = Game.CurrentLevel;
        m_CachedCoachFood = Game.CurrentFoodAmount;
        BrainHUD?.UpdateBrain(frame);
    }

    private static float[] ExtractActionScores(BrainHUDData frame)
    {
        var scores = new float[RogueObservationBuilder.ActionCount];
        if (frame == null || frame.actionRanking == null)
        {
            return scores;
        }

        foreach (var score in frame.actionRanking)
        {
            if (score == null || score.action < 0 || score.action >= scores.Length)
            {
                continue;
            }

            scores[score.action] = score.rolloutScore;
        }

        return scores;
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
            Game.UIManager.SetAgentModeStatus(
                m_AgentMode,
                m_Model != null && m_Model.IsLoaded,
                UseSidecar && m_SidecarOnline);
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

    private void EnsureCoachLogger()
    {
        if (m_CoachLogger == null)
        {
            m_CoachLogger = GetComponent<CoachSessionLogger>();
            if (m_CoachLogger == null)
            {
                m_CoachLogger = gameObject.AddComponent<CoachSessionLogger>();
            }
        }
    }

    private void EnsureCoachLoggingSession()
    {
        if (m_CoachLogRunActive || Game == null || Game.CurrentLevel <= 0 || Game.IsGameOver)
        {
            return;
        }

        EnsureCoachLogger();
        m_CoachLogger.NewSession();
        m_CoachLogger.NewEpisode();
        m_CoachLogRunActive = true;
    }

    private void BindGameEvents()
    {
        if (m_GameEventsBound || Game == null)
        {
            return;
        }

        Game.RunStarted += OnRunStarted;
        Game.RunEnded += OnRunEnded;
        m_GameEventsBound = true;
    }

    private void UnbindGameEvents()
    {
        if (!m_GameEventsBound || Game == null)
        {
            return;
        }

        Game.RunStarted -= OnRunStarted;
        Game.RunEnded -= OnRunEnded;
        m_GameEventsBound = false;
    }

    private void BindPlayerEvents()
    {
        if (Game == null || Game.PlayerController == null)
        {
            return;
        }

        if (m_PlayerEventsBound && ReferenceEquals(m_BoundPlayerController, Game.PlayerController))
        {
            return;
        }

        UnbindPlayerEvents();
        m_BoundPlayerController = Game.PlayerController;
        m_BoundPlayerController.HumanActionStarted += OnHumanActionStarted;
        m_BoundPlayerController.HumanActionFinished += OnHumanActionFinished;
        m_PlayerEventsBound = true;
    }

    private void UnbindPlayerEvents()
    {
        if (!m_PlayerEventsBound || m_BoundPlayerController == null)
        {
            return;
        }

        m_BoundPlayerController.HumanActionStarted -= OnHumanActionStarted;
        m_BoundPlayerController.HumanActionFinished -= OnHumanActionFinished;
        m_BoundPlayerController = null;
        m_PlayerEventsBound = false;
    }

    private void OnRunStarted()
    {
        HandleRunStarted();
    }

    private void HandleRunStarted()
    {
        m_RunInitialized = true;
        m_CoachLogRunActive = false;
        m_PendingRunEndedEvent = null;
        m_PendingTransition = null;
        InvalidateSidecarRequests();
        TransitionRecorder?.BeginEpisode();
        if (m_AgentMode == AgentMode.CoachMode)
        {
            EnsureCoachLoggingSession();
        }

        m_NextDecisionTime = Time.time;
        PublishForCurrentMode();
    }

    private void OnRunEnded(GameManager.RunEndedEvent runEndedEvent)
    {
        m_PendingRunEndedEvent = runEndedEvent;
        FinalizePendingRunEndedIfReady();
    }

    private void FinalizePendingRunEndedIfReady()
    {
        if (m_PendingRunEndedEvent == null || m_PendingTransition != null)
        {
            return;
        }

        if (m_CoachLogRunActive)
        {
            m_CoachLogger?.LogEpisodeEnd(
                m_PendingRunEndedEvent.LevelsCleared,
                m_PendingRunEndedEvent.Died,
                m_PendingRunEndedEvent.FoodRemaining);
        }

        m_PendingRunEndedEvent = null;
    }

    private void OnHumanActionStarted(PlayerController.HumanActionEvent actionEvent)
    {
        if (m_AgentMode != AgentMode.CoachMode || Game == null || Game.IsGameOver)
        {
            return;
        }

        EnsureCoachLoggingSession();
        var frame = GetCoachFrameForCurrentState();
        var scores = ExtractActionScores(frame);
        var snapshot = SnapshotPreStep();
        snapshot.action = actionEvent.ActionIndex;
        snapshot.mode = AgentMode.CoachMode;
        snapshot.suggestedAction = SelectedAction(frame);
        snapshot.actionScores = scores;
        snapshot.riskLevel = BrainPlanner.RiskLevelForAction(Game, actionEvent.ActionIndex);
        snapshot.logCoachAnalytics = true;
        m_PendingTransition = snapshot;
        UpdateCoachFrame(frame);
    }

    private void OnHumanActionFinished(PlayerController.HumanActionEvent actionEvent)
    {
        if (m_AgentMode != AgentMode.CoachMode || m_PendingTransition == null)
        {
            return;
        }

        m_PendingTransition.accepted = actionEvent.Accepted;
        if (!actionEvent.Accepted)
        {
            FlushPendingTransition();
        }
    }

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
            UpdateUi();
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

        UpdateUi();
    }

    private void OnSidecarProbeFailed(string err)
    {
        if (m_SidecarOnline)
        {
            Debug.LogWarning($"Sidecar /info probe failed: {err}. Falling back to mission planner.");
        }

        m_SidecarOnline = false;
        m_SidecarInfoLoaded = false;
        UpdateUi();
    }

    private void RequestCoachSidecarSuggestion()
    {
        if (Game == null || Game.IsGameOver || Game.PlayerController == null || Game.PlayerController.IsMoving)
        {
            return;
        }

        m_SidecarRequestInFlight = true;
        var snapshot = SnapshotPreStep();
        var context = CreateSidecarRequestContext(AgentMode.CoachMode, null);
        int requestHorizon = Mathf.Clamp(PlanningHorizon, 1, Mathf.Max(1, m_SidecarMaxHorizon));
        var request = BuildSidecarRequest(requestHorizon, snapshot);

        StartCoroutine(SidecarClient.PlanActions(
            request,
            response => OnCoachPlanReceived(response, context),
            err => OnCoachPlanFailed(err, context)));
    }

    private void RequestAutonomousSidecarStep()
    {
        if (Game == null || Game.IsGameOver || Game.PlayerController == null || Game.PlayerController.IsMoving)
        {
            return;
        }

        m_SidecarRequestInFlight = true;
        var snapshot = SnapshotPreStep();
        var context = CreateSidecarRequestContext(AgentMode.AIAutonomous, snapshot);
        int requestHorizon = Mathf.Clamp(PlanningHorizon, 1, Mathf.Max(1, m_SidecarMaxHorizon));
        var request = BuildSidecarRequest(requestHorizon, snapshot);

        StartCoroutine(SidecarClient.PlanActions(
            request,
            response => OnAutonomousPlanReceived(response, context),
            err => OnAutonomousPlanFailed(err, context)));
    }

    private LewmClient.PlanRequest BuildSidecarRequest(int requestHorizon, PendingTransition snapshot)
    {
        if (snapshot == null)
        {
            snapshot = SnapshotPreStep();
        }

        return new LewmClient.PlanRequest
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
    }

    private SidecarRequestContext CreateSidecarRequestContext(AgentMode mode, PendingTransition snapshot)
    {
        return new SidecarRequestContext
        {
            Version = ++m_SidecarRequestVersion,
            Mode = mode,
            PlayerCell = Game.PlayerCellPosition,
            Level = Game.CurrentLevel,
            Food = Game.CurrentFoodAmount,
            Snapshot = snapshot,
        };
    }

    private bool IsStaleSidecarContext(SidecarRequestContext context)
    {
        return Game == null
            || context == null
            || context.Version != m_SidecarRequestVersion
            || context.Mode != m_AgentMode
            || Game.IsGameOver
            || Game.CurrentLevel != context.Level
            || Game.CurrentFoodAmount != context.Food
            || Game.PlayerCellPosition != context.PlayerCell;
    }

    private void OnCoachPlanReceived(LewmClient.PlanResponse response, SidecarRequestContext context)
    {
        m_SidecarRequestInFlight = false;
        if (IsStaleSidecarContext(context))
        {
            return;
        }

        m_NextDecisionTime = Time.time + DecisionIntervalSeconds;
        var summary = BrainPlanner.SidecarPlanSummary.FromPlanResponse(response);
        if (summary == null)
        {
            Debug.LogWarning("Sidecar /plan_actions returned malformed plan; falling back to mission planner.");
            m_SidecarOnline = false;
            UpdateCoachFrame(BuildLocalBrainFrame(AgentMode.CoachMode));
            UpdateUi();
            return;
        }

        UpdateCoachFrame(BuildSidecarBrainFrame(summary, AgentMode.CoachMode));
    }

    private void OnCoachPlanFailed(string err, SidecarRequestContext context)
    {
        m_SidecarRequestInFlight = false;
        m_SidecarOnline = false;
        m_NextHealthProbe = Time.time + 0.5f;
        if (IsStaleSidecarContext(context))
        {
            UpdateUi();
            return;
        }

        Debug.LogWarning($"Sidecar /plan_actions failed: {err}. Falling back to mission planner for coach suggestion.");
        m_NextDecisionTime = Time.time + DecisionIntervalSeconds;
        UpdateCoachFrame(BuildLocalBrainFrame(AgentMode.CoachMode));
        UpdateUi();
    }

    private void OnAutonomousPlanReceived(LewmClient.PlanResponse response, SidecarRequestContext context)
    {
        m_SidecarRequestInFlight = false;
        if (IsStaleSidecarContext(context))
        {
            return;
        }

        m_NextDecisionTime = Time.time + DecisionIntervalSeconds;
        var summary = BrainPlanner.SidecarPlanSummary.FromPlanResponse(response);
        if (summary == null)
        {
            Debug.LogWarning("Sidecar /plan_actions returned malformed plan; falling back to mission planner.");
            m_SidecarOnline = false;
            FallbackAutonomousStep(context);
            UpdateUi();
            return;
        }

        var brainFrame = BuildSidecarBrainFrame(summary, AgentMode.AIAutonomous);
        BrainHUD?.UpdateBrain(brainFrame);
        StepWithAction(SelectedAction(brainFrame), context.Snapshot);
    }

    private void OnAutonomousPlanFailed(string err, SidecarRequestContext context)
    {
        m_SidecarRequestInFlight = false;
        m_SidecarOnline = false;
        m_NextHealthProbe = Time.time + 0.5f;
        if (IsStaleSidecarContext(context))
        {
            UpdateUi();
            return;
        }

        m_NextDecisionTime = Time.time + DecisionIntervalSeconds;
        Debug.LogWarning($"Sidecar /plan_actions failed: {err}. Falling back to mission planner for this step.");
        FallbackAutonomousStep(context);
        UpdateUi();
    }

    private void FallbackAutonomousStep(SidecarRequestContext context)
    {
        if (IsStaleSidecarContext(context))
        {
            return;
        }

        var brainFrame = BuildLocalBrainFrame(AgentMode.AIAutonomous);
        BrainHUD?.UpdateBrain(brainFrame);
        StepWithAction(SelectedAction(brainFrame), context.Snapshot);
    }

    private void InvalidateSidecarRequests()
    {
        m_SidecarRequestVersion++;
        m_SidecarRequestInFlight = false;
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

    private void StepWithAction(int action, PendingTransition snapshot)
    {
        if (Game == null || Game.PlayerController == null || action < 0)
        {
            return;
        }

        snapshot.riskLevel = BrainPlanner.RiskLevelForAction(Game, action);
        bool accepted = Game.PlayerController.TryStep(
            RogueObservationBuilder.ActionToDirection(action),
            smoothMovement: !InstantActions);

        if (RecordPlannerTransitions && TransitionRecorder != null)
        {
            snapshot.action = action;
            snapshot.accepted = accepted;
            snapshot.mode = AgentMode.AIAutonomous;
            m_PendingTransition = snapshot;
            if (!accepted || InstantActions)
            {
                FlushPendingTransition();
            }
        }
    }

    private void DecideAndStepViaBrainPlanner()
    {
        var brainFrame = BuildLocalBrainFrame(AgentMode.AIAutonomous);
        BrainHUD?.UpdateBrain(brainFrame);
        var snapshot = SnapshotPreStep();
        StepWithAction(SelectedAction(brainFrame), snapshot);
    }

    private void FlushPendingTransition()
    {
        if (m_PendingTransition == null || Game == null)
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

        if (TransitionRecorder != null)
        {
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
        }

        if (m_PendingTransition.logCoachAnalytics)
        {
            m_CoachLogger?.LogStep(
                m_PendingTransition.suggestedAction,
                m_PendingTransition.action,
                m_PendingTransition.actionScores,
                reward,
                Game.CurrentFoodAmount,
                Game.CurrentLevel);
        }

        Game.RecordDemoActionAttempt(
            m_PendingTransition.mode,
            m_PendingTransition.accepted,
            followedAi: m_PendingTransition.suggestedAction == m_PendingTransition.action,
            coachStep: m_PendingTransition.mode == AgentMode.CoachMode,
            risky: IsRiskyLevel(m_PendingTransition.riskLevel));

        m_PendingTransition = null;

        if (m_AgentMode == AgentMode.CoachMode && !Game.IsGameOver)
        {
            m_NextDecisionTime = Time.time;
            UpdateCoachFrame(BuildLocalBrainFrame(AgentMode.CoachMode));
        }
    }

    private static int SelectedAction(BrainHUDData brainFrame)
    {
        if (brainFrame == null || brainFrame.actionRanking == null || brainFrame.actionRanking.Length == 0)
        {
            return -1;
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

    private static AgentMode NextMode(AgentMode mode)
    {
        return mode switch
        {
            AgentMode.HumanOnly => AgentMode.CoachMode,
            AgentMode.CoachMode => AgentMode.AIAutonomous,
            _ => AgentMode.HumanOnly,
        };
    }

    private static bool IsRiskyLevel(string riskLevel)
    {
        return riskLevel == "medium"
            || riskLevel == "high"
            || riskLevel == "blocked";
    }

    private static float[] CopyScores(float[] scores)
    {
        if (scores == null)
        {
            return null;
        }

        var copy = new float[scores.Length];
        Array.Copy(scores, copy, scores.Length);
        return copy;
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
        public AgentMode mode;
        public int suggestedAction = -1;
        public float[] actionScores;
        public string riskLevel = "low";
        public bool logCoachAnalytics;
    }

    private class SidecarRequestContext
    {
        public int Version;
        public AgentMode Mode;
        public Vector2Int PlayerCell;
        public int Level;
        public int Food;
        public PendingTransition Snapshot;
    }
}
