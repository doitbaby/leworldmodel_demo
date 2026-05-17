using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UIElements;

public class GameManager : MonoBehaviour
{
    private int m_CurrentLevel = 0;
    private bool m_IsInitialized;

    public static GameManager Instance { get; private set; }

    public BoardManager BoardManager;
    public UIManager UIManager;
    public PlayerController PlayerController;
    public UIDocument UIDoc;
    public Key RestartKey = Key.R;

    public Vector2Int PlayerCellPosition => PlayerController.CellPosition;
    public int CurrentLevel => m_CurrentLevel;
    public int CurrentFoodAmount => m_CurrentFoodAmount;

    public TickManager TickManager { get; private set; }

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
        StartNewGame();
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
        m_CurrentLevel = 0;
        m_CurrentFoodAmount = FoodAmount;
        PlayerController.Init();
        SetPlayerInputEnabled(PlayerController.EnableHumanInput);
        UIManager?.HideGameOverPanel();
        NewLevel();
        UIManager?.UpdateFood(FoodAmount);
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
        IsGameOver = true;
        SetPlayerInputEnabled(false);
        if (PlayerController.EnableHumanInput)
        {
            PlayerController.StartNewGameAction.Enable();
        }
        UIManager?.ShowGameOverPanel(m_CurrentLevel);
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

        m_IsInitialized = true;
    }
}
