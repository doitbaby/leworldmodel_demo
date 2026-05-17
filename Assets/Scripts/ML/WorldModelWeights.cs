using System;
using UnityEngine;

public class WorldModelWeights
{
    private ModelData m_Data;

    public bool IsLoaded => m_Data != null && m_Data.obs_size > 0 && m_Data.action_count > 0 && m_Data.hidden_size > 0;

    public int ObservationSize => IsLoaded ? m_Data.obs_size : 0;
    public int ActionCount => IsLoaded ? m_Data.action_count : 0;
    private bool HasSecondLayer => m_Data != null
        && m_Data.hidden_size2 > 0
        && m_Data.w2 != null
        && m_Data.b2 != null;

    public static WorldModelWeights FromJson(string json)
    {
        var weights = new WorldModelWeights();
        weights.m_Data = JsonUtility.FromJson<ModelData>(json);
        weights.Validate();
        return weights;
    }

    public float PredictReward(float[] observation, int action)
    {
        var hidden = Hidden(observation, action);
        return Dot(m_Data.w_reward, hidden, hidden.Length, 0) + m_Data.b_reward[0];
    }

    public float[] PredictNextObservation(float[] observation, int action)
    {
        var hidden = Hidden(observation, action);
        var output = new float[m_Data.obs_size];
        for (int i = 0; i < output.Length; i++)
        {
            output[i] = Dot(m_Data.w_next, hidden, hidden.Length, i * hidden.Length) + m_Data.b_next[i];
        }

        return output;
    }

    private float[] Hidden(float[] observation, int action)
    {
        if (!IsLoaded)
        {
            throw new InvalidOperationException("World model weights are not loaded.");
        }

        if (observation == null || observation.Length != m_Data.obs_size)
        {
            throw new ArgumentException($"Observation must have {m_Data.obs_size} values.");
        }

        action = Mathf.Clamp(action, 0, m_Data.action_count - 1);
        int inputSize = m_Data.obs_size + m_Data.action_count;
        var hidden = new float[m_Data.hidden_size];

        for (int h = 0; h < m_Data.hidden_size; h++)
        {
            float value = m_Data.b1[h];
            int rowOffset = h * inputSize;

            for (int i = 0; i < m_Data.obs_size; i++)
            {
                value += m_Data.w1[rowOffset + i] * observation[i];
            }

            value += m_Data.w1[rowOffset + m_Data.obs_size + action];
            hidden[h] = Mathf.Max(0f, value);
        }

        if (!HasSecondLayer)
        {
            return hidden;
        }

        var hidden2 = new float[m_Data.hidden_size2];
        for (int h = 0; h < m_Data.hidden_size2; h++)
        {
            float value = m_Data.b2[h];
            int rowOffset = h * m_Data.hidden_size;

            for (int i = 0; i < m_Data.hidden_size; i++)
            {
                value += m_Data.w2[rowOffset + i] * hidden[i];
            }

            hidden2[h] = Mathf.Max(0f, value);
        }

        return hidden2;
    }

    private void Validate()
    {
        if (m_Data == null)
        {
            throw new InvalidOperationException("World model JSON did not contain model data.");
        }

        int inputSize = m_Data.obs_size + m_Data.action_count;
        RequireLength(m_Data.w1, m_Data.hidden_size * inputSize, "w1");
        RequireLength(m_Data.b1, m_Data.hidden_size, "b1");

        int finalHiddenSize = m_Data.hidden_size;
        if (HasSecondLayer)
        {
            RequireLength(m_Data.w2, m_Data.hidden_size2 * m_Data.hidden_size, "w2");
            RequireLength(m_Data.b2, m_Data.hidden_size2, "b2");
            finalHiddenSize = m_Data.hidden_size2;
        }

        RequireLength(m_Data.w_next, m_Data.obs_size * finalHiddenSize, "w_next");
        RequireLength(m_Data.b_next, m_Data.obs_size, "b_next");
        RequireLength(m_Data.w_reward, finalHiddenSize, "w_reward");
        RequireLength(m_Data.b_reward, 1, "b_reward");
    }

    private static float Dot(float[] weights, float[] values, int length, int offset)
    {
        float result = 0f;
        for (int i = 0; i < length; i++)
        {
            result += weights[offset + i] * values[i];
        }

        return result;
    }

    private static void RequireLength(float[] values, int expected, string field)
    {
        if (values == null || values.Length != expected)
        {
            throw new InvalidOperationException($"World model field {field} expected {expected} values.");
        }
    }

    [Serializable]
    private class ModelData
    {
        public int obs_size;
        public int action_count;
        public int hidden_size;
        public int hidden_size2;
        public float[] w1;
        public float[] b1;
        public float[] w2;
        public float[] b2;
        public float[] w_next;
        public float[] b_next;
        public float[] w_reward;
        public float[] b_reward;
    }
}
