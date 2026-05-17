using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

public static class RogueDemoSceneBuilder
{
    private const string SourceScene = "Assets/Scenes/Main.unity";
    private const string TrainingScene = "Assets/Scenes/TrainingRoom.unity";
    private const string PlannerScene = "Assets/Scenes/WorldModelPlannerRoom.unity";

    [MenuItem("World Model Demo/Create All Demo Scenes")]
    public static void CreateAllDemoScenes()
    {
        CreateTrainingRoomScene();
        CreatePlannerScene();
    }

    [MenuItem("World Model Demo/Create TrainingRoom Scene")]
    public static void CreateTrainingRoomScene()
    {
        OpenSourceScene();
        var game = ConfigureSharedRoom();
        var playerObject = game.PlayerController.gameObject;

        var recorder = GetOrAdd<RogueTransitionRecorder>(playerObject);
        recorder.RecordTransitions = true;

        var agent = GetOrAdd<RogueAgent>(playerObject);
        agent.Game = game;
        agent.TransitionRecorder = recorder;
        agent.InstantActions = true;
        agent.MaxEpisodeSteps = 160;

        var behavior = GetOrAdd<BehaviorParameters>(playerObject);
        behavior.BehaviorName = "RogueAgent";
        behavior.BehaviorType = BehaviorType.Default;
        behavior.BrainParameters.VectorObservationSize = RogueObservationBuilder.ObservationSize;
        behavior.BrainParameters.NumStackedVectorObservations = 1;
        behavior.BrainParameters.ActionSpec = ActionSpec.MakeDiscrete(RogueObservationBuilder.ActionCount);

        var requester = GetOrAdd<DecisionRequester>(playerObject);
        requester.DecisionPeriod = 1;
        requester.TakeActionsBetweenDecisions = true;

        SaveActiveSceneAs(TrainingScene);
    }

    [MenuItem("World Model Demo/Create Planner Scene")]
    public static void CreatePlannerScene()
    {
        OpenSourceScene();
        var game = ConfigureSharedRoom();
        var playerObject = game.PlayerController.gameObject;

        var planner = GetOrAdd<WorldModelPlannerAgent>(playerObject);
        var brainHud = GetOrAdd<BrainHUDController>(playerObject);
        brainHud.Game = game;
        planner.Game = game;
        planner.BrainHUD = brainHud;
        planner.InstantActions = false;
        planner.DecisionIntervalSeconds = 0.45f;
        planner.PlanningHorizon = 3;
        planner.RolloutDiscount = 0.85f;
        planner.HeuristicBlend = 0.05f;
        planner.StartNewGameOnStart = true;
        planner.StartWithModelEnabled = false;
        planner.DisableHumanInput = false;
        planner.ToggleKey = UnityEngine.InputSystem.Key.M;

        SaveActiveSceneAs(PlannerScene);
    }

    private static GameManager ConfigureSharedRoom()
    {
        var game = Object.FindFirstObjectByType<GameManager>();
        if (game == null)
        {
            throw new MissingReferenceException("Main scene does not contain a GameManager.");
        }

        game.PlayerController.EnableHumanInput = false;
        game.FoodAmount = 40;
        game.BoardManager.Width = 8;
        game.BoardManager.Height = 8;
        game.BoardManager.MinFoodCount = 2;
        game.BoardManager.MaxFoodCount = 4;
        game.BoardManager.MinObstacleCount = 1;
        game.BoardManager.MaxObstacleCount = 2;
        game.BoardManager.MinEnemyCount = 0;
        game.BoardManager.MaxEnemyCount = 1;

        EditorUtility.SetDirty(game);
        EditorUtility.SetDirty(game.PlayerController);
        EditorUtility.SetDirty(game.BoardManager);
        return game;
    }

    private static void OpenSourceScene()
    {
        EditorSceneManager.OpenScene(SourceScene, OpenSceneMode.Single);
    }

    private static void SaveActiveSceneAs(string path)
    {
        EditorSceneManager.SaveScene(SceneManager.GetActiveScene(), path);
        AssetDatabase.SaveAssets();
        Debug.Log($"World Model demo scene saved to {path}");
    }

    private static T GetOrAdd<T>(GameObject target) where T : Component
    {
        var component = target.GetComponent<T>();
        if (component == null)
        {
            component = target.AddComponent<T>();
            EditorUtility.SetDirty(target);
        }

        EditorUtility.SetDirty(component);
        return component;
    }
}
