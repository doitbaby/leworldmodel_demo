using System;

[Serializable]
public class DemoRunSummary
{
    public AgentMode mode = AgentMode.HumanOnly;
    public bool mixedMode;
    public int activeSeed;
    public int levelReached;
    public int foodRemaining;
    public int totalActionAttempts;
    public int invalidMoves;
    public int riskyMoves;
    public int followedAiCount;
    public int totalCoachSteps;
    public float complianceRate;

    public string ModeLabel => mixedMode ? "MIXED" : mode.ToString();

    public DemoRunSummary Clone()
    {
        return (DemoRunSummary)MemberwiseClone();
    }
}
