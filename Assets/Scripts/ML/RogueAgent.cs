using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.InputSystem;

public class RogueAgent : Agent
{
    [Header("Scene References")]
    public GameManager Game;
    public RogueTransitionRecorder TransitionRecorder;

    [Header("Episode")]
    public int MaxEpisodeSteps = 160;
    public bool InstantActions = true;

    [Header("Reward")]
    public float StepPenalty = 0.001f;
    public float InvalidActionPenalty = 0.025f;
    public float ProgressReward = 0.05f;
    public float FoodDeltaReward = 0.01f;
    public float ExitReward = 1.0f;
    public float GameOverPenalty = -1.0f;
    public float MaxStepsPenalty = -0.25f;

    private int m_EpisodeSteps;

    public override void Initialize()
    {
        EnsureReferences();
    }

    public override void OnEpisodeBegin()
    {
        EnsureReferences();
        if (Game == null)
        {
            return;
        }

        Game.PlayerController.EnableHumanInput = false;
        Game.StartNewGame();
        Game.SetPlayerInputEnabled(false);
        TransitionRecorder?.BeginEpisode();
        m_EpisodeSteps = 0;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        RogueObservationBuilder.WriteToSensor(sensor, Game);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        EnsureReferences();
        if (Game == null || Game.IsGameOver)
        {
            AddReward(GameOverPenalty);
            EndEpisode();
            return;
        }

        int action = actions.DiscreteActions[0];
        var previousObservation = RogueObservationBuilder.Build(Game);
        int previousBoardWidth = PixelObservationBuilder.GetBoardWidth(Game);
        int previousBoardHeight = PixelObservationBuilder.GetBoardHeight(Game);
        int[] previousBoardState = PixelObservationBuilder.BuildCellCodes(Game);
        int previousLevel = Game.CurrentLevel;
        int previousFood = Game.CurrentFoodAmount;
        int previousDistance = Game.DistanceToExit(Game.PlayerCellPosition);

        bool accepted = Game.PlayerController.TryStep(
            RogueObservationBuilder.ActionToDirection(action),
            smoothMovement: !InstantActions);

        bool reachedExit = Game.CurrentLevel > previousLevel;
        int currentDistance = reachedExit ? 0 : Game.DistanceToExit(Game.PlayerCellPosition);
        float reward = -StepPenalty;

        if (!accepted)
        {
            reward -= InvalidActionPenalty;
        }

        reward += (previousDistance - currentDistance) * ProgressReward;
        reward += (Game.CurrentFoodAmount - previousFood) * FoodDeltaReward;

        bool done = reachedExit || Game.IsGameOver;
        string outcome = "running";

        if (reachedExit)
        {
            reward += ExitReward;
            outcome = "exit";
        }
        else if (Game.IsGameOver)
        {
            reward += GameOverPenalty;
            outcome = "game_over";
        }

        m_EpisodeSteps++;
        if (!done && m_EpisodeSteps >= MaxEpisodeSteps)
        {
            reward += MaxStepsPenalty;
            done = true;
            outcome = "max_steps";
        }

        AddReward(reward);

        var nextObservation = RogueObservationBuilder.Build(Game);
        if (TransitionRecorder != null)
        {
            TransitionRecorder.Record(
                previousObservation,
                action,
                reward,
                nextObservation,
                done,
                Game.CurrentLevel,
                Game.CurrentFoodAmount,
                outcome,
                previousBoardWidth,
                previousBoardHeight,
                previousBoardState,
                PixelObservationBuilder.BuildCellCodes(Game));
        }

        if (done)
        {
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discrete = actionsOut.DiscreteActions;
        discrete[0] = 0;

        var keyboard = Keyboard.current;
        if (keyboard == null)
        {
            return;
        }

        if (keyboard.sKey.isPressed || keyboard.downArrowKey.isPressed)
        {
            discrete[0] = 1;
        }
        else if (keyboard.aKey.isPressed || keyboard.leftArrowKey.isPressed)
        {
            discrete[0] = 2;
        }
        else if (keyboard.dKey.isPressed || keyboard.rightArrowKey.isPressed)
        {
            discrete[0] = 3;
        }
    }

    private void EnsureReferences()
    {
        if (Game == null)
        {
            Game = GameManager.Instance != null
                ? GameManager.Instance
                : FindAnyObjectByType<GameManager>();
        }

        if (TransitionRecorder == null)
        {
            TransitionRecorder = GetComponent<RogueTransitionRecorder>();
        }
    }
}
