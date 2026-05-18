using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// Lightweight HTTP client for the Python LeWorldModel inference sidecar.
///
/// Talks to the FastAPI app in <c>tools/lewm/sidecar.py</c>:
///   GET  /healthz
///   GET  /info
///   POST /score_actions     { board_width, board_height, board_state,
///                             horizon, num_sequences,
///                             action_sequences_flat, discount, done_penalty }
///
/// Designed as a passive utility. M3 lands the wire; M5 plugs it into
/// <c>BrainPlanner</c> / <c>WorldModelPlannerAgent</c> so the planner
/// can score candidate plans through the sidecar with a graceful
/// fallback to the existing mission heuristic + LeWM-lite path when the
/// sidecar is unreachable.
///
/// Inheriting from <see cref="MonoBehaviour"/> so the component can be
/// added to a GameObject in the inspector and can drive its own
/// coroutines. Usage:
///
/// <code>
/// var client = gameObject.AddComponent&lt;LewmClient&gt;();
/// client.BaseUrl = "http://127.0.0.1:5555";
/// var req = new LewmClient.ScoreRequest { ... };
/// StartCoroutine(client.ScoreActions(req, OnScore, OnFail));
/// </code>
/// </summary>
public class LewmClient : MonoBehaviour
{
    [Tooltip("Base URL of the Python sidecar (default: localhost:5555).")]
    public string BaseUrl = "http://127.0.0.1:5555";

    [Tooltip("Per-request timeout in seconds. Demo decision interval is ~0.45s.")]
    [Range(0.05f, 5f)]
    public float TimeoutSeconds = 0.5f;

    public IEnumerator Healthz(Action<HealthzResponse> onSuccess, Action<string> onFailure = null)
    {
        return SendGet<HealthzResponse>("/healthz", onSuccess, onFailure);
    }

    public IEnumerator Info(Action<InfoResponse> onSuccess, Action<string> onFailure = null)
    {
        return SendGet<InfoResponse>("/info", onSuccess, onFailure);
    }

    public IEnumerator ScoreActions(
        ScoreRequest request,
        Action<ScoreResponse> onSuccess,
        Action<string> onFailure = null)
    {
        if (request == null)
        {
            onFailure?.Invoke("ScoreRequest is null");
            yield break;
        }

        string body = JsonUtility.ToJson(request);
        byte[] payload = Encoding.UTF8.GetBytes(body);

        using (var www = new UnityWebRequest(BaseUrl + "/score_actions", UnityWebRequest.kHttpVerbPOST))
        {
            www.uploadHandler = new UploadHandlerRaw(payload);
            www.downloadHandler = new DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");
            www.timeout = Mathf.Max(1, Mathf.CeilToInt(TimeoutSeconds));

            yield return www.SendWebRequest();

            if (www.result != UnityWebRequest.Result.Success)
            {
                onFailure?.Invoke($"/score_actions HTTP {www.responseCode}: {www.error}");
                yield break;
            }

            ScoreResponse parsed = null;
            try
            {
                parsed = JsonUtility.FromJson<ScoreResponse>(www.downloadHandler.text);
            }
            catch (Exception ex)
            {
                onFailure?.Invoke("/score_actions parse error: " + ex.Message);
                yield break;
            }

            if (parsed == null || parsed.scores == null)
            {
                onFailure?.Invoke("/score_actions parsed null/scores");
                yield break;
            }
            onSuccess?.Invoke(parsed);
        }
    }

    private IEnumerator SendGet<T>(string path, Action<T> onSuccess, Action<string> onFailure)
        where T : class
    {
        using (var www = UnityWebRequest.Get(BaseUrl + path))
        {
            www.timeout = Mathf.Max(1, Mathf.CeilToInt(TimeoutSeconds));
            yield return www.SendWebRequest();
            if (www.result != UnityWebRequest.Result.Success)
            {
                onFailure?.Invoke($"{path} HTTP {www.responseCode}: {www.error}");
                yield break;
            }
            T parsed = null;
            try
            {
                parsed = JsonUtility.FromJson<T>(www.downloadHandler.text);
            }
            catch (Exception ex)
            {
                onFailure?.Invoke($"{path} parse error: " + ex.Message);
                yield break;
            }
            onSuccess?.Invoke(parsed);
        }
    }

    [Serializable]
    public class ScoreRequest
    {
        public int board_width;
        public int board_height;
        public int[] board_state;
        public int horizon;
        public int num_sequences;
        public int[] action_sequences_flat;
        public float discount = 0.95f;
        public float done_penalty = 1.0f;
    }

    [Serializable]
    public class ScoreResponse
    {
        public float[] scores;
        public int horizon;
        public string service;
    }

    [Serializable]
    public class InfoResponse
    {
        public string service;
        public int embed_dim;
        public int action_dim;
        public int image_size;
        public int sequence_length;
    }

    [Serializable]
    public class HealthzResponse
    {
        public string status;
        public string service;
    }
}
