using UnityEngine;
using UnityEngine.InputSystem;

public class PlayerController : MonoBehaviour
{
    private BoardManager m_Board;
    public Vector2Int CellPosition { get; set; }
    public bool IsMoving => m_IsMoving;
    public bool EnableHumanInput = true;

    public InputAction MoveAction;
    public InputAction StartNewGameAction;

    public float MoveSpeed = 5.0f;

    private bool m_IsMoving;
    private Vector3 m_MoveTarget;

    private Animator m_Animator;
    private int Animator_Moving = Animator.StringToHash("Moving");
    private int Animator_Attack = Animator.StringToHash("Attack");

    void Awake()
    {
        m_Animator = GetComponent<Animator>();
    }

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
    }

    // Update is called once per frame
    void Update()
    {
        if (GameManager.Instance.IsGameOver)
        {
            if (EnableHumanInput && StartNewGameAction.triggered)
            {
                GameManager.Instance.StartNewGame();
            }
            return;
        }

        if (m_IsMoving)
        {
            transform.position = Vector3.MoveTowards(transform.position, m_MoveTarget, MoveSpeed * Time.deltaTime);

            if (transform.position == m_MoveTarget)
            {
                CompleteMove();
            }
            return;
        }

        if (EnableHumanInput && MoveAction.triggered)
        {
            Vector2 move = MoveAction.ReadValue<Vector2>();
            TryStep(Vector2Int.RoundToInt(move));
        }
    }

    public void Init()
    {
        m_IsMoving = false;
    }

    /// <summary>
    /// Puts the character on the board at the given position.
    /// </summary>
    public void Spawn(BoardManager boardManager, Vector2Int cell)
    {
        m_Board = boardManager;
        MoveTo(cell, smoothMovement: false);
    }

    public void MoveTo(Vector2Int cell, bool smoothMovement = true)
    {
        //technically the player is not there yet, but the movement is only cosmetic
        //and we know nothing can stop it as we checked everything before starting it
        //so safe to update there!
        CellPosition = cell;

        if (smoothMovement)
        {
            m_IsMoving = true;
            m_MoveTarget = m_Board.CellToWorld(CellPosition);
        }
        else
        {
            m_IsMoving = false;
            transform.position = m_Board.CellToWorld(CellPosition);
        }

        if (m_Animator != null)
        {
            m_Animator.SetBool(Animator_Moving, m_IsMoving);
        }

        if (!smoothMovement)
        {
            CompleteMove();
        }
    }

    public bool TryStep(Vector2Int direction, bool smoothMovement = true)
    {
        if (m_Board == null || m_IsMoving || GameManager.Instance.IsGameOver)
        {
            return false;
        }

        direction.x = Mathf.Clamp(direction.x, -1, 1);
        direction.y = Mathf.Clamp(direction.y, -1, 1);

        if (Mathf.Abs(direction.x) + Mathf.Abs(direction.y) != 1)
        {
            return false;
        }

        var newCellTarget = CellPosition + direction;
        var cellData = m_Board.GetCellData(newCellTarget);
        if (cellData == null || !cellData.Passable)
        {
            return false;
        }

        GameManager.Instance.TickManager.Tick();
        if (GameManager.Instance.IsGameOver)
        {
            return true;
        }

        if (cellData.ContainedObject == null)
        {
            MoveTo(newCellTarget, smoothMovement);
            return true;
        }

        var canEnter = cellData.ContainedObject.PlayerWantsToEnter();
        if (canEnter)
        {
            MoveTo(newCellTarget, smoothMovement);
        }
        else
        {
            Attack();
        }

        return true;
    }

    public void Attack()
    {
        if (m_Animator != null)
        {
            m_Animator.SetTrigger(Animator_Attack);
        }
    }

    private void CompleteMove()
    {
        m_IsMoving = false;

        if (m_Animator != null)
        {
            m_Animator.SetBool(Animator_Moving, false);
        }

        var cellData = m_Board.GetCellData(CellPosition);
        if (cellData != null && cellData.ContainedObject != null)
        {
            cellData.ContainedObject.PlayerEntered();
        }
    }
}
