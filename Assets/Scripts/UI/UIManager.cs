using System;
using UnityEngine.UIElements;

public class UIManager
{
    public UIDocument UIDoc;

    private Label m_LevelLabel;
    private Label m_FoodLabel;
    private Label m_ModelStatusLabel;
    private Button m_ModelToggleButton;
    private Button m_RestartButton;
    private Action m_ModelToggleAction;
    private Action m_RestartAction;

    private VisualElement m_GameOverPanel;
    private Label m_GameOverMessage;

    public void UpdateLevel(int level) => m_LevelLabel.text = $" - LEVEL {level} -";

    public void UpdateFood(int food) => m_FoodLabel.text = $"Food : {food}";

    public void RegisterWorldModelToggle(Action callback)
    {
        if (m_ModelToggleButton == null)
        {
            return;
        }

        if (m_ModelToggleAction != null)
        {
            m_ModelToggleButton.clicked -= m_ModelToggleAction;
        }

        m_ModelToggleAction = callback;
        if (m_ModelToggleAction != null)
        {
            m_ModelToggleButton.clicked += m_ModelToggleAction;
        }
    }

    public void RegisterRestart(Action callback)
    {
        if (m_RestartButton == null)
        {
            return;
        }

        if (m_RestartAction != null)
        {
            m_RestartButton.clicked -= m_RestartAction;
        }

        m_RestartAction = callback;
        if (m_RestartAction != null)
        {
            m_RestartButton.clicked += m_RestartAction;
        }
    }

    public void SetWorldModelStatus(bool modelControlEnabled, bool modelLoaded)
    {
        SetAgentModeStatus(
            modelControlEnabled ? AgentMode.AIAutonomous : AgentMode.HumanOnly,
            modelLoaded,
            false);
    }

    public void SetAgentModeStatus(AgentMode mode, bool modelLoaded, bool sidecarOnline)
    {
        if (m_ModelToggleButton == null || m_ModelStatusLabel == null)
        {
            return;
        }

        m_ModelToggleButton.text = mode switch
        {
            AgentMode.CoachMode => "COACH",
            AgentMode.AIAutonomous => "AI",
            _ => "HUMAN",
        };

        m_ModelStatusLabel.text = mode switch
        {
            AgentMode.CoachMode => sidecarOnline
                ? "COACH+SIDECAR"
                : (modelLoaded ? "COACH+MODEL" : "COACH+HEUR"),
            AgentMode.AIAutonomous => sidecarOnline
                ? "SIDECAR AI"
                : (modelLoaded ? "WORLD MODEL" : "HEURISTIC AI"),
            _ => "PLAYER",
        };
    }

    public void ShowGameOverPanel(int level)
    {
        m_GameOverMessage.text = $"GAME OVER!\n\nYou traveled through {level} levels";
        m_GameOverPanel.style.visibility = Visibility.Visible;
    }

    public void HideGameOverPanel() => m_GameOverPanel.style.visibility = Visibility.Hidden;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    public void Init(UIDocument uiDoc)
    {
        UIDoc = uiDoc;
        m_LevelLabel = UIDoc.rootVisualElement.Q<Label>("LevelLabel");
        m_FoodLabel = UIDoc.rootVisualElement.Q<Label>("FoodLabel");
        m_ModelStatusLabel = UIDoc.rootVisualElement.Q<Label>("ModelStatusLabel");
        m_ModelToggleButton = UIDoc.rootVisualElement.Q<Button>("ModelToggleButton");
        m_GameOverPanel = UIDoc.rootVisualElement.Q<VisualElement>("GameOverPanel");
        m_GameOverMessage = m_GameOverPanel.Q<Label>("GameOverMessage");
        m_RestartButton = UIDoc.rootVisualElement.Q<Button>("RestartButton");
        HideGameOverPanel();
    }
}
