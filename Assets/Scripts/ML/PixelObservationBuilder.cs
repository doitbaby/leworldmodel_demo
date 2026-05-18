using UnityEngine;

/// <summary>
/// Builds the discrete cell-code grid that the LeWorldModel port consumes
/// as a pixel observation.
///
/// Rather than capturing the rendered scene with a Camera/RenderTexture,
/// we serialize a small <c>(Width * Height)</c> int array of cell codes.
/// The Python side (see <c>tools/lewm/data.py::render_board_to_pixels</c>)
/// turns these into a deterministic RGB tile image, which keeps the JSONL
/// transition file small, avoids lighting / animation noise during
/// training, and makes the encoding identical to the synthetic data used
/// by <c>SyntheticRogueDataset</c>.
///
/// Cell code conventions (must stay in lock-step with
/// <c>ROGUE_CELL_COLORS</c> in <c>tools/lewm/data.py</c>):
///   -1 wall, 0 empty, 1 exit, 2 enemy, 3 obstacle, 4 food, 5 player.
/// </summary>
public static class PixelObservationBuilder
{
    public const int CellWall = -1;
    public const int CellEmpty = 0;
    public const int CellExit = 1;
    public const int CellEnemy = 2;
    public const int CellObstacle = 3;
    public const int CellFood = 4;
    public const int CellPlayer = 5;

    /// <summary>
    /// Returns a flat row-major <c>(width * height)</c> int array of cell
    /// codes, with the player overlaid on its current cell. The board is
    /// addressed as <c>codes[y * width + x]</c>.
    /// </summary>
    public static int[] BuildCellCodes(GameManager gameManager)
    {
        if (gameManager == null || gameManager.BoardManager == null)
        {
            return new int[0];
        }

        var board = gameManager.BoardManager;
        int w = board.Width;
        int h = board.Height;

        if (w <= 0 || h <= 0)
        {
            return new int[0];
        }

        var codes = new int[w * h];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                codes[y * w + x] = board.GetCellObjectCode(new Vector2Int(x, y));
            }
        }

        if (gameManager.PlayerController != null)
        {
            var playerCell = gameManager.PlayerCellPosition;
            if (playerCell.x >= 0 && playerCell.x < w
                && playerCell.y >= 0 && playerCell.y < h)
            {
                codes[playerCell.y * w + playerCell.x] = CellPlayer;
            }
        }

        return codes;
    }

    public static int GetBoardWidth(GameManager gameManager)
    {
        if (gameManager == null || gameManager.BoardManager == null)
        {
            return 0;
        }
        return gameManager.BoardManager.Width;
    }

    public static int GetBoardHeight(GameManager gameManager)
    {
        if (gameManager == null || gameManager.BoardManager == null)
        {
            return 0;
        }
        return gameManager.BoardManager.Height;
    }
}
