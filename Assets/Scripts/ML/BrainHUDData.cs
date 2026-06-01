using System;

[Serializable]
public class BrainHUDData
{
    public bool modelLoaded;
    public bool aiEnabled;
    public AgentMode agentMode;
    public string modeLabel;
    public string selectedAction;
    public string explanation;
    public bool isCoachMode;
    public int suggestedActionIndex;
    public string suggestedActionName;
    public float complianceRate;
    public int sessionFollowedSteps;
    public int sessionTotalSteps;
    public string riskWarning;
    public BrainActionScore[] actionRanking;
    public BrainFuture[] imaginedFutures;
    public BrainMetrics metrics;
}

[Serializable]
public class BrainActionScore
{
    public int action;
    public string actionName;
    public float predictedReward;
    public float rolloutScore;
    public bool selected;
    public string reason;
}

[Serializable]
public class BrainFuture
{
    public string[] actions;
    public float score;
    public string summary;
}

[Serializable]
public class BrainMetrics
{
    public string schema = "rogue.metrics.v1";
    public string source = "not_loaded";
    public int transitionCount;
    public int epochs;
    public float dynamicsLoss;
    public float rewardLoss;
    public float valLoss;
    public float successRate;

    public static BrainMetrics Empty()
    {
        return new BrainMetrics
        {
            source = "not_loaded",
            transitionCount = 0,
            epochs = 0,
            dynamicsLoss = 0f,
            rewardLoss = 0f,
            valLoss = 0f,
            successRate = 0f,
        };
    }
}
