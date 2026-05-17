using System.Collections.Generic;
using Unity.MLAgents.Sensors;
using UnityEngine;

public static class RogueObservationBuilder
{
    public const int ObservationSize = 31;
    public const int ActionCount = 4;

    private static readonly Vector2Int[] Directions =
    {
        Vector2Int.up,
        Vector2Int.down,
        Vector2Int.left,
        Vector2Int.right,
    };

    public static float[] Build(GameManager gameManager)
    {
        var values = new List<float>(ObservationSize);

        if (gameManager == null || gameManager.BoardManager == null || gameManager.PlayerController == null)
        {
            return EmptyObservation();
        }

        var board = gameManager.BoardManager;
        var playerCell = gameManager.PlayerCellPosition;
        var exitCell = board.ExitCell;
        float widthScale = Mathf.Max(1, board.Width - 1);
        float heightScale = Mathf.Max(1, board.Height - 1);
        float maxDistance = Mathf.Max(1, board.Width + board.Height);
        int currentDistance = gameManager.DistanceToExit(playerCell);

        values.Add(playerCell.x / widthScale);
        values.Add(playerCell.y / heightScale);
        values.Add((exitCell.x - playerCell.x) / widthScale);
        values.Add((exitCell.y - playerCell.y) / heightScale);
        values.Add(currentDistance / maxDistance);
        values.Add(gameManager.FoodAmount <= 0 ? 0f : gameManager.CurrentFoodAmount / (float)gameManager.FoodAmount);
        values.Add(gameManager.IsGameOver ? 1f : 0f);
        values.Add(Mathf.Clamp01(gameManager.CurrentLevel / 10f));

        foreach (var direction in Directions)
        {
            var targetCell = playerCell + direction;
            var data = board.GetCellData(targetCell);
            int objectCode = board.GetCellObjectCode(targetCell);
            bool passable = data != null && data.Passable;
            int targetDistance = Mathf.Abs(exitCell.x - targetCell.x) + Mathf.Abs(exitCell.y - targetCell.y);

            values.Add(passable ? 1f : 0f);
            values.Add(objectCode / 5f);
            values.Add(targetDistance < currentDistance ? 1f : 0f);
            values.Add(objectCode == 2 ? 1f : 0f);
        }

        if (board.TryFindNearestEnemy(playerCell, out var enemyCell, out int enemyDistance))
        {
            values.Add((enemyCell.x - playerCell.x) / widthScale);
            values.Add((enemyCell.y - playerCell.y) / heightScale);
            values.Add(enemyDistance / maxDistance);
        }
        else
        {
            values.Add(0f);
            values.Add(0f);
            values.Add(1f);
        }

        foreach (var direction in Directions)
        {
            var targetCell = playerCell + direction;
            int targetDistance = Mathf.Abs(exitCell.x - targetCell.x) + Mathf.Abs(exitCell.y - targetCell.y);
            values.Add((currentDistance - targetDistance) / maxDistance);
        }

        while (values.Count < ObservationSize)
        {
            values.Add(0f);
        }

        if (values.Count > ObservationSize)
        {
            values.RemoveRange(ObservationSize, values.Count - ObservationSize);
        }

        return values.ToArray();
    }

    public static void WriteToSensor(VectorSensor sensor, GameManager gameManager)
    {
        var values = Build(gameManager);
        for (int i = 0; i < values.Length; i++)
        {
            sensor.AddObservation(values[i]);
        }
    }

    public static Vector2Int ActionToDirection(int action)
    {
        action = Mathf.Clamp(action, 0, Directions.Length - 1);
        return Directions[action];
    }

    public static string ActionName(int action)
    {
        switch (Mathf.Clamp(action, 0, Directions.Length - 1))
        {
            case 0:
                return "up";
            case 1:
                return "down";
            case 2:
                return "left";
            default:
                return "right";
        }
    }

    private static float[] EmptyObservation()
    {
        return new float[ObservationSize];
    }
}
