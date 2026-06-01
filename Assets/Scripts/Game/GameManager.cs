using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UIElements;

public class GameManager : MonoBehaviour
{
    public class RunEndedEvent
    {
        public int LevelsCleared;
        public bool Died;
        public int FoodRemaining;
    }

    private int m_CurrentLevel = 0;
    private bool m_IsInitialized;
    private bool m_PlayerEventsBound;
    private bool m_DemoRunBaselineStored;
    private int m_DemoRunIndex;
    private AgentMode m_CurrentAgentMode = AgentMode.HumanOnly;

    public static GameManager Instance { get; private set; }

    public BoardManager BoardManager;
    public UIManager UIManager;
    public PlayerController PlayerController;
    public UIDocument UIDoc;
    public Key RestartKey = Key.R;
    public bool UseFixedDemoSeed = true;
    public int DemoSeed = 20260601;
    public bool ReplaySameSeed = true;

    public Vector2Int PlayerCellPosition => PlayerController.CellPosition;
    public int CurrentLevel => m_CurrentLevel;
    public int CurrentFoodAmount => m_CurrentFoodAmount;
    public DemoRunSummary CurrentRunSummary { get; private set; }
    public DemoRunSummary PreviousHumanOnlySummary { get; private set; }

    public TickManager TickManager { get; private set; }
    public event System.Action RunStarted;
    public event System.Action<RunEndedEvent> RunEnded;

    public int FoodAmount = 100;
    private int m_CurrentFoodAmount;

    public bool IsGameOver { get; set; } = false;

    public CellData Cell { get; set; }

    private void Awake()
    {
        if (Instance != null)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
    }

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        EnsureInitialized();
        if (m_CurrentLevel == 0 && !IsGameOver)
        {
            StartNewGame();
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (IsGameOver && Keyboard.current != null && Keyboard.current[RestartKey].wasPressedThisFrame)
        {
            StartNewGame();
        }
    }

    public void StartNewGame()
    {
        EnsureInitialized();
        IsGameOver = false;
        m_DemoRunBaselineStored = false;
        m_CurrentLevel = 0;
        m_CurrentFoodAmount = FoodAmount;
        BeginDemoRunSummary();
        PlayerController.Init();
        SetPlayerInputEnabled(PlayerController.EnableHumanInput);
        UIManager?.HideGameOverPanel();
        NewLevel();
        UIManager?.UpdateFood(FoodAmount);
        RunStarted?.Invoke();
    }

    public void NewLevel()
    {
        BoardManager.Clean();
        BoardManager.Init();
        PlayerController.Spawn(BoardManager, new Vector2Int(1, 1));
        m_CurrentLevel++;
        UIManager?.UpdateLevel(m_CurrentLevel);
    }

    void OnTickHappen()
    {
        ChangeFood(-1);
    }

    public void ChangeFood(int amount)
    {
        m_CurrentFoodAmount += amount;
        UIManager?.UpdateFood(m_CurrentFoodAmount);
        Debug.Log("Current amount of food : " + m_CurrentFoodAmount);
        if (m_CurrentFoodAmount <= 0)
        {
            GameOver();
        }
    }

    public void GameOver()
    {
        if (IsGameOver)
        {
            return;
        }

        IsGameOver = true;
        SetPlayerInputEnabled(false);
        if (PlayerController.EnableHumanInput)
        {
            PlayerController.StartNewGameAction.Enable();
        }

        FinalizeDemoRunSummary();
        RunEnded?.Invoke(new RunEndedEvent
        {
            LevelsCleared = Mathf.Max(0, m_CurrentLevel),
            Died = true,
            FoodRemaining = m_CurrentFoodAmount,
        });

        ShowDemoRunSummary(storeBaseline: false);
    }

    public void SetPlayerInputEnabled(bool enabled)
    {
        if (enabled)
        {
            PlayerController.MoveAction.Enable();
            PlayerController.StartNewGameAction.Disable();
        }
        else
        {
            PlayerController.MoveAction.Disable();
            PlayerController.StartNewGameAction.Disable();
        }
    }

    public void SetDemoRunMode(AgentMode mode)
    {
        if (CurrentRunSummary != null
            && !IsGameOver
            && CurrentRunSummary.mode != mode)
        {
            CurrentRunSummary.mixedMode = true;
        }

        if (CurrentRunSummary != null && !CurrentRunSummary.mixedMode)
        {
            CurrentRunSummary.mode = mode;
        }

        m_CurrentAgentMode = mode;
    }

    public void RecordDemoActionAttempt(
        AgentMode mode,
        bool accepted,
        bool followedAi,
        bool coachStep,
        bool risky)
    {
        if (CurrentRunSummary == null)
        {
            return;
        }

        CurrentRunSummary.totalActionAttempts++;
        if (!accepted)
        {
            CurrentRunSummary.invalidMoves++;
        }

        if (risky)
        {
            CurrentRunSummary.riskyMoves++;
        }

        if (coachStep)
        {
            CurrentRunSummary.totalCoachSteps++;
            if (followedAi)
            {
                CurrentRunSummary.followedAiCount++;
            }

            CurrentRunSummary.complianceRate = CurrentRunSummary.totalCoachSteps > 0
                ? (float)CurrentRunSummary.followedAiCount / CurrentRunSummary.totalCoachSteps
                : 0f;
        }

        CurrentRunSummary.foodRemaining = m_CurrentFoodAmount;
        CurrentRunSummary.levelReached = m_CurrentLevel;
        if (IsGameOver)
        {
            ShowDemoRunSummary(storeBaseline: true);
        }
    }

    private void OnHumanActionFinished(PlayerController.HumanActionEvent actionEvent)
    {
        if (m_CurrentAgentMode != AgentMode.HumanOnly)
        {
            return;
        }

        RecordDemoActionAttempt(
            AgentMode.HumanOnly,
            actionEvent.Accepted,
            followedAi: false,
            coachStep: false,
            risky: false);
    }

    private void BeginDemoRunSummary()
    {
        int activeSeed = UseFixedDemoSeed
            ? (ReplaySameSeed ? DemoSeed : DemoSeed + m_DemoRunIndex)
            : 0;
        m_DemoRunIndex++;

        if (UseFixedDemoSeed)
        {
            Random.InitState(activeSeed);
        }

        CurrentRunSummary = new DemoRunSummary
        {
            mode = m_CurrentAgentMode,
            activeSeed = activeSeed,
            levelReached = 0,
            foodRemaining = FoodAmount,
        };
    }

    private void FinalizeDemoRunSummary()
    {
        if (CurrentRunSummary == null)
        {
            return;
        }

        CurrentRunSummary.levelReached = m_CurrentLevel;
        CurrentRunSummary.foodRemaining = m_CurrentFoodAmount;
        CurrentRunSummary.complianceRate = CurrentRunSummary.totalCoachSteps > 0
            ? (float)CurrentRunSummary.followedAiCount / CurrentRunSummary.totalCoachSteps
            : 0f;
    }

    private static bool IsPureHumanOnlyRun(DemoRunSummary summary)
    {
        return summary != null
            && !summary.mixedMode
            && summary.mode == AgentMode.HumanOnly;
    }

    private void ShowDemoRunSummary(bool storeBaseline)
    {
        var previousHumanOnly = PreviousHumanOnlySummary;
        UIManager?.ShowGameOverPanel(m_CurrentLevel, CurrentRunSummary, previousHumanOnly);
        if (storeBaseline && !m_DemoRunBaselineStored && IsPureHumanOnlyRun(CurrentRunSummary))
        {
            PreviousHumanOnlySummary = CurrentRunSummary.Clone();
            m_DemoRunBaselineStored = true;
        }
    }

    public int DistanceToExit(Vector2Int cell)
    {
        var exitCell = BoardManager.ExitCell;
        return Mathf.Abs(exitCell.x - cell.x) + Mathf.Abs(exitCell.y - cell.y);
    }

    public void EnsureInitialized()
    {
        if (m_IsInitialized)
        {
            return;
        }

        TickManager = new TickManager();
        TickManager.OnTick += OnTickHappen;

        if (UIDoc != null)
        {
            UIManager = new UIManager();
            UIManager.Init(UIDoc);
            UIManager.RegisterRestart(StartNewGame);
        }

        if (PlayerController != null && !m_PlayerEventsBound)
        {
            PlayerController.HumanActionFinished += OnHumanActionFinished;
            m_PlayerEventsBound = true;
        }

        m_IsInitialized = true;
    }

    private void OnDestroy()
    {
        if (PlayerController != null && m_PlayerEventsBound)
        {
            PlayerController.HumanActionFinished -= OnHumanActionFinished;
        }
    }
}
