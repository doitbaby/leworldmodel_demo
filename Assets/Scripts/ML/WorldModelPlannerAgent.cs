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

    private WorldModelWeights m_Model;
    private BrainMetrics m_Metrics = BrainMetrics.Empty();
    private float m_NextDecisionTime;
    private bool m_IsModelControlEnabled;
    private bool m_UiBound;
    private PendingTransition m_PendingTransition;

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

        m_NextDecisionTime = Time.time + DecisionIntervalSeconds;
        var brainFrame = BuildBrainFrame();
        BrainHUD?.UpdateBrain(brainFrame);
        int action = SelectedAction(brainFrame);
        var previousObservation = RogueObservationBuilder.Build(Game);
        int previousLevel = Game.CurrentLevel;
        int previousFood = Game.CurrentFoodAmount;
        int previousDistance = Game.DistanceToExit(Game.PlayerCellPosition);
        int previousBoardWidth = PixelObservationBuilder.GetBoardWidth(Game);
        int previousBoardHeight = PixelObservationBuilder.GetBoardHeight(Game);
        int[] previousBoardState = PixelObservationBuilder.BuildCellCodes(Game);
        bool accepted = Game.PlayerController.TryStep(RogueObservationBuilder.ActionToDirection(action), smoothMovement: !InstantActions);

        if (RecordPlannerTransitions && TransitionRecorder != null)
        {
            m_PendingTransition = new PendingTransition
            {
                observation = previousObservation,
                action = action,
                accepted = accepted,
                previousLevel = previousLevel,
                previousFood = previousFood,
                previousDistance = previousDistance,
                previousBoardWidth = previousBoardWidth,
                previousBoardHeight = previousBoardHeight,
                previousBoardState = previousBoardState,
            };

            if (!accepted || InstantActions)
            {
                FlushPendingTransition();
            }
        }
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
