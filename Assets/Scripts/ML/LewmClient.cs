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
///   POST /plan_actions      { board_width, board_height, board_state,
///                             horizon, num_candidates, top_k,
///                             discount, done_penalty, seed }   (M5)
///
/// M3 introduced this component as a passive utility; M5 wires it into
/// <see cref="WorldModelPlannerAgent"/> so the brain queries
/// <c>/plan_actions</c> as the primary scorer with a graceful fallback
/// to the existing mission heuristic when the sidecar is unreachable.
///
/// Inheriting from <see cref="MonoBehaviour"/> so the component can be
/// added to a GameObject in the inspector and can drive its own
/// coroutines. Usage:
///
/// <code>
/// var client = gameObject.AddComponent&lt;LewmClient&gt;();
/// client.BaseUrl = "http://127.0.0.1:5555";
/// var req = new LewmClient.PlanRequest { ... };
/// StartCoroutine(client.PlanActions(req, OnPlan, OnFail));
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
        return SendPost<ScoreRequest, ScoreResponse>("/score_actions", request, onSuccess, onFailure);
    }

    public IEnumerator PlanActions(
        PlanRequest request,
        Action<PlanResponse> onSuccess,
        Action<string> onFailure = null)
    {
        return SendPost<PlanRequest, PlanResponse>("/plan_actions", request, onSuccess, onFailure);
    }

    private IEnumerator SendPost<TReq, TResp>(
        string path,
        TReq request,
        Action<TResp> onSuccess,
        Action<string> onFailure)
        where TReq : class
        where TResp : class
    {
        if (request == null)
        {
            onFailure?.Invoke($"{path} request is null");
            yield break;
        }

        string body = JsonUtility.ToJson(request);
        byte[] payload = Encoding.UTF8.GetBytes(body);

        using (var www = new UnityWebRequest(BaseUrl + path, UnityWebRequest.kHttpVerbPOST))
        {
            www.uploadHandler = new UploadHandlerRaw(payload);
            www.downloadHandler = new DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");
            www.timeout = Mathf.Max(1, Mathf.CeilToInt(TimeoutSeconds));

            yield return www.SendWebRequest();

            if (www.result != UnityWebRequest.Result.Success)
            {
                onFailure?.Invoke($"{path} HTTP {www.responseCode}: {www.error}");
                yield break;
            }

            TResp parsed = null;
            try
            {
                parsed = JsonUtility.FromJson<TResp>(www.downloadHandler.text);
            }
            catch (Exception ex)
            {
                onFailure?.Invoke($"{path} parse error: " + ex.Message);
                yield break;
            }

            if (parsed == null)
            {
                onFailure?.Invoke($"{path} parsed null");
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

    /// <summary>
    /// Body for <c>POST /plan_actions</c>. The sidecar samples
    /// <see cref="num_candidates"/> random action sequences server-side,
    /// scores them through the JEPA, and returns the best plus a top-k
    /// summary. Defaults match the Python <c>PlanRequest</c> dataclass.
    /// </summary>
    [Serializable]
    public class PlanRequest
    {
        public int board_width;
        public int board_height;
        public int[] board_state;
        public int horizon;
        public int num_candidates = 64;
        public int top_k = 3;
        public float discount = 0.95f;
        public float done_penalty = 1.0f;
        // 0 = "let sidecar pick" (sidecar treats 0 as no seed).
        public int seed;
    }

    /// <summary>
    /// Body for <c>POST /plan_actions</c> responses. The server flattens
    /// <c>top_k_actions</c> row-major to a single int[] because Unity
    /// JsonUtility cannot deserialize nested arrays. Use
    /// <see cref="GetTopKAction"/> to read the k-th sequence's t-th step.
    /// </summary>
    [Serializable]
    public class PlanResponse
    {
        public int[] best_actions;
        public float best_score;
        public int[] top_k_actions_flat;
        public float[] top_k_scores;
        public int num_candidates;
        public int top_k;
        public int horizon;
        public string service;

        public int GetTopKAction(int k, int t)
        {
            if (top_k_actions_flat == null || horizon <= 0)
            {
                return -1;
            }
            int index = k * horizon + t;
            if (index < 0 || index >= top_k_actions_flat.Length)
            {
                return -1;
            }
            return top_k_actions_flat[index];
        }
    }

    [Serializable]
    public class InfoResponse
    {
        public string service;
        public int embed_dim;
        public int action_dim;
        public int image_size;
        public int sequence_length;
        public int max_horizon;
    }

    [Serializable]
    public class HealthzResponse
    {
        public string status;
        public string service;
    }
}
