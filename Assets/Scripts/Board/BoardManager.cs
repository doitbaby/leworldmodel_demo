using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.Tilemaps;

public class BoardManager : MonoBehaviour
{
    private Tilemap m_Tilemap;
    private CellData[,] m_BoardData;
    private Grid m_Grid;
    private List<Vector2Int> m_EmptyCellsList;

    public int Width;
    public int Height;
    public Tile[] GroundTiles;
    public Tile[] WallTiles;
    // public PlayerController PlayerController;

    public FoodCellObject[] FoodPrefabs;
    public ObstacleCellObject[] ObstaclePrefabs;
    public EnemyCellObject[] EnemyPrefabs;
    public ExitCellObject ExitCellPrefab;

    public int MinFoodCount = 3;
    public int MaxFoodCount = 7;

    public int MinObstacleCount = 4;
    public int MaxObstacleCount = 8;

    public int MinEnemyCount = 1;
    public int MaxEnemyCount = 2;
    public Vector2Int ExitCell => new Vector2Int(Width - 2, Height - 2);

    public void Init()
    {
        m_Tilemap = GetComponentInChildren<Tilemap>();
        m_Grid = GetComponentInChildren<Grid>();
        m_EmptyCellsList = new List<Vector2Int>();

        m_BoardData = new CellData[Width, Height];

        for (int y = 0; y < Height; ++y)
        {
            for (int x = 0; x < Width; ++x)
            {
                Tile tile;
                m_BoardData[x, y] = new CellData();

                if (x == 0 || y == 0 || x == Width - 1 || y == Height - 1)
                {
                    tile = WallTiles[Random.Range(0, WallTiles.Length)];
                    m_BoardData[x, y].Passable = false;
                }
                else
                {
                    tile = GroundTiles[Random.Range(0, GroundTiles.Length)];
                    m_BoardData[x, y].Passable = true;
                    m_EmptyCellsList.Add(new Vector2Int(x, y));
                }

                m_Tilemap.SetTile(new Vector3Int(x, y, 0), tile);
            }
        }

        var playerStartingPosition = new Vector2Int(1, 1);
        m_EmptyCellsList.Remove(playerStartingPosition);

        Vector2Int endCoord = ExitCell;
        AddObject(Instantiate(ExitCellPrefab), endCoord);
        m_EmptyCellsList.Remove(endCoord);

        GenerateObstacles();
        GenerateFood();
        GenerateEnemies();
    }

    public void Clean()
    {
        //no board data, so exit early, nothing to clean
        if (m_BoardData == null)
            return;

        for (int y = 0; y < Height; ++y)
        {
            for (int x = 0; x < Width; ++x)
            {
                var cellData = m_BoardData[x, y];
                if (cellData.ContainedObject != null)
                {
                    //CAREFUL! Destroy the GameObject NOT just cellData.ContainedObject
                    //Otherwise what you are destroying is the JUST CellObject COMPONENT
                    //and not the whole gameobject with sprite
                    Destroy(cellData.ContainedObject.gameObject);
                }

                m_Tilemap.SetTile(new Vector3Int(x, y, 0), null);
            }
        }
    }

    public Vector3 CellToWorld(Vector2Int cellIndex) => m_Grid.GetCellCenterWorld((Vector3Int)cellIndex);

    public CellData GetCellData(Vector2Int cellIndex)
    {
        if (m_BoardData == null
            || cellIndex.x < 0 || cellIndex.x >= Width
            || cellIndex.y < 0 || cellIndex.y >= Height)
            return null;
        return m_BoardData[cellIndex.x, cellIndex.y];
    }

    public int GetCellObjectCode(Vector2Int cellIndex)
    {
        var data = GetCellData(cellIndex);
        if (data == null || !data.Passable)
        {
            return -1;
        }

        if (data.ContainedObject == null)
        {
            return 0;
        }

        if (data.ContainedObject is ExitCellObject)
        {
            return 1;
        }

        if (data.ContainedObject is EnemyCellObject)
        {
            return 2;
        }

        if (data.ContainedObject is ObstacleCellObject)
        {
            return 3;
        }

        if (data.ContainedObject is FoodCellObject)
        {
            return 4;
        }

        return 5;
    }

    public bool TryFindNearestEnemy(Vector2Int origin, out Vector2Int enemyCell, out int distance)
    {
        enemyCell = Vector2Int.zero;
        distance = int.MaxValue;

        if (m_BoardData == null)
        {
            return false;
        }

        for (int y = 0; y < Height; ++y)
        {
            for (int x = 0; x < Width; ++x)
            {
                var data = m_BoardData[x, y];
                if (data.ContainedObject is EnemyCellObject)
                {
                    int candidateDistance = Mathf.Abs(origin.x - x) + Mathf.Abs(origin.y - y);
                    if (candidateDistance < distance)
                    {
                        distance = candidateDistance;
                        enemyCell = new Vector2Int(x, y);
                    }
                }
            }
        }

        return distance != int.MaxValue;
    }

    void GenerateFood()
    {
        int cnt = Random.Range(MinFoodCount, MaxFoodCount + 1);
        for (int i = 0; i < cnt; ++i)
        {
            int randomIndex = Random.Range(0, m_EmptyCellsList.Count);
            Vector2Int coord = m_EmptyCellsList[randomIndex];
            m_EmptyCellsList.RemoveAt(randomIndex);
            CellData data = m_BoardData[coord.x, coord.y];
            var newObj = Instantiate(FoodPrefabs[Random.Range(0, FoodPrefabs.Length)]);
            AddObject(newObj, coord);
        }
    }

    void GenerateEnemies()
    {
        int cnt = Random.Range(MinEnemyCount, MaxEnemyCount + 1);
        for (int i = 0; i < cnt; ++i)
        {
            int randomIndex = Random.Range(0, m_EmptyCellsList.Count);
            Vector2Int coord = m_EmptyCellsList[randomIndex];
            m_EmptyCellsList.RemoveAt(randomIndex);
            CellData data = m_BoardData[coord.x, coord.y];
            var newObj = Instantiate(EnemyPrefabs[Random.Range(0, EnemyPrefabs.Length)]);
            AddObject(newObj, coord);
        }
    }

    void GenerateObstacles()
    {
        int cnt = Random.Range(MinObstacleCount, MaxObstacleCount + 1);
        for (int i = 0; i < cnt; ++i)
        {
            int randomIndex = Random.Range(0, m_EmptyCellsList.Count);
            Vector2Int coord = m_EmptyCellsList[randomIndex];

            m_EmptyCellsList.RemoveAt(randomIndex);
            CellData data = m_BoardData[coord.x, coord.y];
            ObstacleCellObject newObj = Instantiate(ObstaclePrefabs[Random.Range(0, ObstaclePrefabs.Length)]);
            AddObject(newObj, coord);
        }
    }

    void AddObject(CellObject obj, Vector2Int coord)
    {
        CellData data = m_BoardData[coord.x, coord.y];
        obj.transform.position = CellToWorld(coord);
        data.ContainedObject = obj;
        obj.Init(coord);
    }

    public Tile GetCellTile(Vector2Int cellIndex)
    {
        return m_Tilemap.GetTile<Tile>(new Vector3Int(cellIndex.x, cellIndex.y, 0));
    }

    public void SetCellTile(Vector2Int cellIndex, Tile tile)
    {
        m_Tilemap.SetTile(new Vector3Int(cellIndex.x, cellIndex.y, 0), tile);
    }
}
