using System;
using System.IO;
using UnityEngine;

public class CoachSessionLogger : MonoBehaviour
{
    public int totalSteps { get; private set; }
    public int followedSteps { get; private set; }
    public float complianceRate { get; private set; }

    private string m_SessionId;
    private string m_OutputPath;
    private StreamWriter m_Writer;
    private int m_EpisodeIndex;

    private static readonly string[] ActionNames = { "up", "down", "left", "right" };

    private void OnDestroy()
    {
        CloseWriter();
    }

    private void OnApplicationQuit()
    {
        CloseWriter();
    }

    public void NewSession()
    {
        CloseWriter();
        totalSteps = 0;
        followedSteps = 0;
        complianceRate = 0f;
        m_EpisodeIndex = 0;
        m_SessionId = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");

        string dir = Path.Combine(Application.persistentDataPath, "coach_sessions");
        Directory.CreateDirectory(dir);
        m_OutputPath = Path.Combine(dir, $"session_{m_SessionId}.jsonl");
        m_Writer = new StreamWriter(m_OutputPath, append: false);
        Debug.Log($"[CoachLogger] Session started -> {m_OutputPath}");
    }

    public void NewEpisode()
    {
        m_EpisodeIndex++;
    }

    public void LogStep(
        int suggestedActionIdx,
        int humanActionIdx,
        float[] actionScores,
        float reward,
        int food,
        int level)
    {
        totalSteps++;
        bool followed = suggestedActionIdx == humanActionIdx;
        if (followed)
        {
            followedSteps++;
        }

        complianceRate = totalSteps > 0
            ? (float)followedSteps / totalSteps
            : 0f;

        string scores = actionScores != null
            ? $"[{string.Join(",", Array.ConvertAll(actionScores, score => score.ToString("F4")))}]"
            : "[]";

        string line = "{"
            + $"\"session\":\"{m_SessionId}\","
            + $"\"episode\":{m_EpisodeIndex},"
            + $"\"step\":{totalSteps},"
            + $"\"ai_suggested\":\"{IdxToName(suggestedActionIdx)}\","
            + $"\"human_chosen\":\"{IdxToName(humanActionIdx)}\","
            + $"\"followed\":{(followed ? "true" : "false")},"
            + $"\"scores\":{scores},"
            + $"\"reward\":{reward:F4},"
            + $"\"food\":{food},"
            + $"\"level\":{level}"
            + "}";

        m_Writer?.WriteLine(line);
        m_Writer?.Flush();
    }

    public void LogEpisodeEnd(int levelsCleared, bool died, int totalFood)
    {
        string line = "{"
            + $"\"session\":\"{m_SessionId}\","
            + $"\"episode\":{m_EpisodeIndex},"
            + "\"type\":\"episode_end\","
            + $"\"levels_cleared\":{levelsCleared},"
            + $"\"died\":{(died ? "true" : "false")},"
            + $"\"food_remaining\":{totalFood},"
            + $"\"compliance_rate\":{complianceRate:F4},"
            + $"\"total_steps\":{totalSteps},"
            + $"\"followed_steps\":{followedSteps}"
            + "}";

        m_Writer?.WriteLine(line);
        m_Writer?.Flush();
    }

    private static string IdxToName(int idx)
    {
        return idx >= 0 && idx < ActionNames.Length
            ? ActionNames[idx]
            : "none";
    }

    private void CloseWriter()
    {
        if (m_Writer == null)
        {
            return;
        }

        m_Writer.Flush();
        m_Writer.Close();
        m_Writer = null;

        if (!string.IsNullOrEmpty(m_OutputPath))
        {
            Debug.Log($"[CoachLogger] Session saved -> {m_OutputPath}");
        }
    }
}
