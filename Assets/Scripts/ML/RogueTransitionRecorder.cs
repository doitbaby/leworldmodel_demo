using System;
using System.Globalization;
using System.IO;
using UnityEngine;

public class RogueTransitionRecorder : MonoBehaviour
{
    private const string Schema = "rogue.transition.v2";

    public bool RecordTransitions = true;
    public string OutputDirectoryName = "WorldModelDemo";
    public string OutputFileName = "rogue_transitions.jsonl";

    private int m_Episode;
    private int m_Step;

    public string OutputPath
    {
        get
        {
            return Path.Combine(Application.persistentDataPath, OutputDirectoryName, OutputFileName);
        }
    }

    public void BeginEpisode()
    {
        m_Episode++;
        m_Step = 0;
    }

    public void Record(
        float[] observation,
        int action,
        float reward,
        float[] nextObservation,
        bool done,
        int level,
        int food,
        string outcome)
    {
        if (!RecordTransitions)
        {
            return;
        }

        Directory.CreateDirectory(Path.GetDirectoryName(OutputPath));

        var transition = new RogueTransition
        {
            schema = Schema,
            timestamp = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture),
            episode = Mathf.Max(1, m_Episode),
            step = m_Step,
            obs = observation,
            action = action,
            action_name = RogueObservationBuilder.ActionName(action),
            reward = reward,
            next_obs = nextObservation,
            done = done,
            outcome = outcome,
            level = level,
            food = food,
        };

        File.AppendAllText(OutputPath, JsonUtility.ToJson(transition) + Environment.NewLine);
        m_Step++;
    }

    [Serializable]
    private class RogueTransition
    {
        public string schema;
        public string timestamp;
        public int episode;
        public int step;
        public float[] obs;
        public int action;
        public string action_name;
        public float reward;
        public float[] next_obs;
        public bool done;
        public string outcome;
        public int level;
        public int food;
    }
}
