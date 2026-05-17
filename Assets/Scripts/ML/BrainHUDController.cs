using System.Linq;
using UnityEngine;
using UnityEngine.UIElements;

public class BrainHUDController : MonoBehaviour
{
    public GameManager Game;

    private VisualElement m_Root;
    private VisualElement m_Panel;
    private Label m_ModeLabel;
    private Label m_SelectedActionLabel;
    private Label m_ExplanationLabel;
    private Label m_MetricsLabel;
    private VisualElement m_RankingList;
    private VisualElement m_FuturesList;
    private bool m_IsBound;

    private void Start()
    {
        Bind();
        ShowWaitingState();
    }

    public void UpdateBrain(BrainHUDData data)
    {
        Bind();
        if (!m_IsBound || data == null)
        {
            return;
        }

        m_ModeLabel.text = data.aiEnabled
            ? $"{data.modeLabel} ACTIVE"
            : $"{data.modeLabel} READY";
        m_SelectedActionLabel.text = $"BEST ACTION: {data.selectedAction}";
        m_ExplanationLabel.text = data.explanation;
        UpdateMetrics(data.metrics);
        UpdateRanking(data.actionRanking);
        UpdateFutures(data.imaginedFutures);
    }

    private void Bind()
    {
        if (m_IsBound)
        {
            return;
        }

        if (Game == null)
        {
            Game = GameManager.Instance != null
                ? GameManager.Instance
                : FindAnyObjectByType<GameManager>();
        }

        if (Game == null || Game.UIDoc == null)
        {
            return;
        }

        m_Root = Game.UIDoc.rootVisualElement;
        m_Panel = m_Root.Q<VisualElement>("BrainPanel");
        m_ModeLabel = m_Root.Q<Label>("BrainModeLabel");
        m_SelectedActionLabel = m_Root.Q<Label>("BrainSelectedActionLabel");
        m_ExplanationLabel = m_Root.Q<Label>("BrainExplanationLabel");
        m_MetricsLabel = m_Root.Q<Label>("BrainMetricsLabel");
        m_RankingList = m_Root.Q<VisualElement>("BrainRankingList");
        m_FuturesList = m_Root.Q<VisualElement>("BrainFuturesList");

        m_IsBound = m_Panel != null
            && m_ModeLabel != null
            && m_SelectedActionLabel != null
            && m_ExplanationLabel != null
            && m_MetricsLabel != null
            && m_RankingList != null
            && m_FuturesList != null;
    }

    private void ShowWaitingState()
    {
        if (!m_IsBound)
        {
            return;
        }

        m_ModeLabel.text = "PLAYER CONTROL";
        m_SelectedActionLabel.text = "BEST ACTION: waiting";
        m_ExplanationLabel.text = "Toggle AI to inspect action ranking and imagined futures.";
        m_MetricsLabel.text = "metrics: waiting for model";
        m_RankingList.Clear();
        m_FuturesList.Clear();
    }

    private void UpdateMetrics(BrainMetrics metrics)
    {
        if (metrics == null || metrics.transitionCount == 0)
        {
            m_MetricsLabel.text = "metrics: no training metrics loaded";
            return;
        }

        m_MetricsLabel.text =
            $"source {metrics.source} | transitions {metrics.transitionCount} | dyn {metrics.dynamicsLoss:0.0000} | rew {metrics.rewardLoss:0.0000} | success {metrics.successRate:P0}";
    }

    private void UpdateRanking(BrainActionScore[] ranking)
    {
        m_RankingList.Clear();
        if (ranking == null || ranking.Length == 0)
        {
            AddTextRow(m_RankingList, "No action scores yet.", false);
            return;
        }

        float min = ranking.Min(item => item.rolloutScore);
        float max = ranking.Max(item => item.rolloutScore);
        foreach (var score in ranking)
        {
            AddRankingRow(score, min, max);
        }
    }

    private void UpdateFutures(BrainFuture[] futures)
    {
        m_FuturesList.Clear();
        if (futures == null || futures.Length == 0)
        {
            AddTextRow(m_FuturesList, "No imagined futures yet.", false);
            return;
        }

        for (int i = 0; i < futures.Length; i++)
        {
            var future = futures[i];
            string actionPath = future.actions != null && future.actions.Length > 0
                ? string.Join(" -> ", future.actions.Select(action => action.ToUpperInvariant()))
                : "none";
            AddTextRow(m_FuturesList, $"{i + 1}. {actionPath}  score {future.score:0.00}", i == 0);
        }
    }

    private void AddRankingRow(BrainActionScore score, float minScore, float maxScore)
    {
        var row = new VisualElement();
        row.style.marginBottom = 6;
        row.style.paddingLeft = 6;
        row.style.paddingRight = 6;
        row.style.paddingTop = 5;
        row.style.paddingBottom = 5;
        row.style.backgroundColor = score.selected
            ? new Color(0.12f, 0.35f, 0.18f, 0.88f)
            : new Color(0f, 0f, 0f, 0.35f);

        var label = new Label($"{score.actionName.ToUpperInvariant()}  reward {score.predictedReward:0.00}  future {score.rolloutScore:0.00}");
        label.style.color = Color.white;
        label.style.fontSize = 11;
        label.style.unityTextAlign = TextAnchor.MiddleLeft;
        row.Add(label);

        var barOuter = new VisualElement();
        barOuter.style.height = 7;
        barOuter.style.backgroundColor = new Color(1f, 1f, 1f, 0.18f);
        barOuter.style.marginTop = 4;

        var barInner = new VisualElement();
        float width = Mathf.Approximately(maxScore, minScore)
            ? 50f
            : Mathf.Lerp(8f, 100f, Mathf.InverseLerp(minScore, maxScore, score.rolloutScore));
        barInner.style.width = Length.Percent(width);
        barInner.style.height = 7;
        barInner.style.backgroundColor = score.selected
            ? new Color(0.45f, 0.95f, 0.48f, 1f)
            : new Color(0.38f, 0.65f, 0.95f, 0.9f);
        barOuter.Add(barInner);
        row.Add(barOuter);
        m_RankingList.Add(row);
    }

    private static void AddTextRow(VisualElement parent, string text, bool highlight)
    {
        var label = new Label(text);
        label.style.color = Color.white;
        label.style.fontSize = 11;
        label.style.whiteSpace = WhiteSpace.Normal;
        label.style.marginBottom = 6;
        label.style.paddingLeft = 6;
        label.style.paddingRight = 6;
        label.style.paddingTop = 5;
        label.style.paddingBottom = 5;
        label.style.backgroundColor = highlight
            ? new Color(0.12f, 0.35f, 0.18f, 0.88f)
            : new Color(0f, 0f, 0f, 0.35f);
        parent.Add(label);
    }
}
