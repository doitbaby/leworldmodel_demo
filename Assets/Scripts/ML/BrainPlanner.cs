using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public static class BrainPlanner
{
    private const float InvalidScore = -999f;
    private const float EnemyFirstStepPenalty = -6f;
    private const float ObstacleFirstStepPenalty = -4f;
    private const float ExitFirstStepBonus = 3f;
    private const float FoodFirstStepBonus = 0.25f;
    private const float MissionScoreBase = 40f;
    private const float MissionPathBonus = 4f;
    private const float ModelScoreWeight = 0.35f;
    private const float HeuristicRolloutWeight = 0.12f;
    private const float UnreachablePathCost = 9999f;
    private const int LowFoodThreshold = 30;
    // M5: how much weight the safety bias gets when the sidecar plan is
    // the primary scorer. Small enough that a strong model preference
    // dominates, large enough that walking into an adjacent enemy still
    // loses to a slightly weaker but safer alternative (enemy penalty
    // = -6, so 0.5 * -6 = -3 of bias).
    private const float SidecarSafetyWeight = 0.5f;
    // M5: score reserved for actions that the sidecar's top-k did not
    // sample. We do not want them to compete with the model's choices
    // unless the model failed to differentiate. The map heuristic still
    // tells us if the action is reachable; we just bias it well below
    // any plan that actually got scored.
    private const float UnconsideredActionBase = -50f;

    public static BrainHUDData Decide(
        GameManager game,
        WorldModelWeights model,
        BrainMetrics metrics,
        int planningHorizon,
        float discount,
        float heuristicBlend,
        bool aiEnabled)
    {
        planningHorizon = Mathf.Clamp(planningHorizon, 1, 5);
        discount = Mathf.Clamp01(discount);
        bool modelLoaded = model != null && model.IsLoaded;
        var observation = RogueObservationBuilder.Build(game);
        var rankings = new List<BrainActionScore>();
        var futures = new List<BrainFuture>();
        var missionTarget = ChooseMissionTarget(game);

        for (int action = 0; action < RogueObservationBuilder.ActionCount; action++)
        {
            RolloutResult rollout = modelLoaded
                ? EvaluateModelRollout(model, observation, action, planningHorizon, discount)
                : EvaluateHeuristicRollout(game, action, planningHorizon, discount);

            MissionActionScore missionScore = EvaluateMissionAction(game, action, missionTarget);
            float auxiliaryScore = modelLoaded
                ? rollout.TotalScore * ModelScoreWeight
                : rollout.TotalScore * HeuristicRolloutWeight;
            float rolloutScore = IsInvalidScore(missionScore.Score)
                ? InvalidScore
                : missionScore.Score + auxiliaryScore + FirstStepSafetyScore(game, action);
            var futureActions = BuildMissionFutureActions(game, action, missionTarget, planningHorizon);

            rankings.Add(new BrainActionScore
            {
                action = action,
                actionName = RogueObservationBuilder.ActionName(action),
                predictedReward = rollout.FirstReward,
                rolloutScore = rolloutScore,
                selected = false,
                reason = BuildActionReason(game, action, rolloutScore, missionScore),
            });

            futures.Add(new BrainFuture
            {
                actions = futureActions.Select(RogueObservationBuilder.ActionName).ToArray(),
                score = rolloutScore,
                summary = BuildFutureSummary(futureActions, missionTarget, modelLoaded),
            });
        }

        var orderedRankings = rankings
            .OrderByDescending(score => score.rolloutScore)
            .ToArray();

        if (orderedRankings.Length > 0)
        {
            orderedRankings[0].selected = true;
        }

        var selected = orderedRankings.Length > 0 ? orderedRankings[0] : null;
        return new BrainHUDData
        {
            modelLoaded = modelLoaded,
            aiEnabled = aiEnabled,
            modeLabel = modelLoaded ? "WORLD MODEL + MISSION" : "MISSION FALLBACK",
            selectedAction = selected != null ? selected.actionName.ToUpperInvariant() : "NONE",
            explanation = selected != null
                ? BuildSelectionExplanation(selected, missionTarget, modelLoaded)
                : "No valid action was found.",
            actionRanking = orderedRankings,
            imaginedFutures = futures
                .OrderByDescending(future => future.score)
                .Take(3)
                .ToArray(),
            metrics = metrics ?? BrainMetrics.Empty(),
        };
    }

    /// <summary>
    /// M5 — model-first action selection driven by a sidecar plan
    /// (random-shooting JEPA rollouts).
    ///
    /// Unlike <see cref="Decide"/>, the mission heuristic no longer
    /// dominates the score. The sidecar's best plan is the primary
    /// scorer, the first-step safety bias is a small soft constraint
    /// (e.g. avoid walking adjacent into an enemy when an equally good
    /// alternative exists), and the map heuristic is used purely to
    /// hard-reject blocked / out-of-bounds actions.
    ///
    /// Caller must have already confirmed the sidecar is online and
    /// supplied a valid <paramref name="plan"/>. If the plan is empty or
    /// nothing the model proposed is reachable, falls back to the
    /// mission planner via <see cref="Decide"/>.
    /// </summary>
    public static BrainHUDData DecideWithSidecarPlan(
        GameManager game,
        SidecarPlanSummary plan,
        BrainMetrics metrics,
        bool aiEnabled)
    {
        if (plan == null
            || plan.BestActions == null
            || plan.BestActions.Length == 0
            || plan.TopKActions == null
            || plan.TopKActions.Length == 0)
        {
            // No usable plan; defer to the mission-heuristic path so the
            // HUD still has something to display.
            return Decide(game, null, metrics, 3, 0.85f, 0f, aiEnabled);
        }

        var rankings = new BrainActionScore[RogueObservationBuilder.ActionCount];
        var futures = new List<BrainFuture>(plan.TopKActions.Length);

        // For each cardinal action find the best plan score among the
        // top-k that starts with it. We do NOT re-query the sidecar per
        // action: random-shooting top-k naturally covers a spread of
        // first steps, and one HTTP call per decision is the right
        // budget at the 0.45s decision interval.
        var actionToScore = new Dictionary<int, float>();
        var actionToReason = new Dictionary<int, string>();
        for (int k = 0; k < plan.TopKActions.Length; k++)
        {
            var seq = plan.TopKActions[k];
            if (seq == null || seq.Length == 0)
            {
                continue;
            }
            int firstAction = seq[0];
            float candidate = plan.TopKScores != null && k < plan.TopKScores.Length
                ? plan.TopKScores[k]
                : float.NegativeInfinity;
            if (!actionToScore.TryGetValue(firstAction, out float existing) || candidate > existing)
            {
                actionToScore[firstAction] = candidate;
                actionToReason[firstAction] = k == 0
                    ? $"world-model best plan (score {candidate:0.00})"
                    : $"world-model alt plan top-{k + 1} (score {candidate:0.00})";
            }
        }

        for (int action = 0; action < RogueObservationBuilder.ActionCount; action++)
        {
            float mapScore = MapHeuristicScore(game, action);
            bool blocked = IsInvalidScore(mapScore);
            float modelScore;
            string reason;
            if (blocked)
            {
                modelScore = InvalidScore;
                reason = "blocked or unsafe";
            }
            else if (actionToScore.TryGetValue(action, out float plannedScore))
            {
                modelScore = plannedScore;
                reason = actionToReason[action];
            }
            else
            {
                // Action did not show up in the top-k. Keep it eligible
                // (it is reachable) but score it well below the model's
                // picks so it only wins if all model picks are blocked.
                modelScore = UnconsideredActionBase + mapScore * 0.1f;
                reason = "world model did not consider this action in its top-k";
            }

            float safetyAdj = FirstStepSafetyScore(game, action);
            float finalScore = blocked
                ? InvalidScore
                : modelScore + safetyAdj * SidecarSafetyWeight;

            rankings[action] = new BrainActionScore
            {
                action = action,
                actionName = RogueObservationBuilder.ActionName(action),
                predictedReward = blocked ? 0f : modelScore,
                rolloutScore = finalScore,
                selected = false,
                reason = reason,
            };
        }

        for (int k = 0; k < plan.TopKActions.Length; k++)
        {
            var seq = plan.TopKActions[k];
            if (seq == null || seq.Length == 0)
            {
                continue;
            }
            var names = seq.Select(RogueObservationBuilder.ActionName).ToArray();
            float kScore = plan.TopKScores != null && k < plan.TopKScores.Length
                ? plan.TopKScores[k]
                : 0f;
            futures.Add(new BrainFuture
            {
                actions = names,
                score = kScore,
                summary = $"world-model rollout top-{k + 1} (score {kScore:0.00}): {string.Join(" -> ", names)}",
            });
        }

        var ordered = rankings.OrderByDescending(s => s.rolloutScore).ToArray();
        BrainActionScore selectedScore = null;
        if (ordered.Length > 0)
        {
            ordered[0].selected = true;
            selectedScore = ordered[0];
        }

        string label = $"WORLD MODEL SIDECAR ({plan.NumCandidates}x{plan.Horizon})";
        return new BrainHUDData
        {
            modelLoaded = true,
            aiEnabled = aiEnabled,
            modeLabel = label,
            selectedAction = selectedScore != null
                ? selectedScore.actionName.ToUpperInvariant()
                : "NONE",
            explanation = selectedScore != null
                ? $"Choose {selectedScore.actionName.ToUpperInvariant()} because {selectedScore.reason}."
                : "No valid action was found in the sidecar plan.",
            actionRanking = ordered,
            imaginedFutures = futures
                .OrderByDescending(f => f.score)
                .Take(3)
                .ToArray(),
            metrics = metrics ?? BrainMetrics.Empty(),
        };
    }

    public static float MapHeuristicScore(GameManager game, int action)
    {
        if (game == null || game.BoardManager == null)
        {
            return InvalidScore;
        }

        var direction = RogueObservationBuilder.ActionToDirection(action);
        var target = game.PlayerCellPosition + direction;
        var cell = game.BoardManager.GetCellData(target);
        if (cell == null || !cell.Passable)
        {
            return InvalidScore;
        }

        int currentDistance = game.DistanceToExit(game.PlayerCellPosition);
        int targetDistance = game.DistanceToExit(target);
        float score = currentDistance - targetDistance;

        int objectCode = game.BoardManager.GetCellObjectCode(target);
        if (objectCode == 1)
        {
            score += 10f;
        }
        else if (objectCode == 2)
        {
            score -= 4f;
        }
        else if (objectCode == 3)
        {
            score -= 3f;
        }
        else if (objectCode == 4)
        {
            score += 0.5f;
        }

        return score;
    }

    private static RolloutResult EvaluateModelRollout(
        WorldModelWeights model,
        float[] observation,
        int firstAction,
        int depth,
        float discount)
    {
        float firstReward = model.PredictReward(observation, firstAction);
        var actions = new List<int> { firstAction };

        if (depth <= 1)
        {
            return new RolloutResult(actions, firstReward, firstReward);
        }

        var nextObservation = model.PredictNextObservation(observation, firstAction);
        RolloutResult bestFuture = null;
        for (int action = 0; action < RogueObservationBuilder.ActionCount; action++)
        {
            var candidate = EvaluateModelRollout(model, nextObservation, action, depth - 1, discount);
            if (bestFuture == null || candidate.TotalScore > bestFuture.TotalScore)
            {
                bestFuture = candidate;
            }
        }

        if (bestFuture == null)
        {
            return new RolloutResult(actions, firstReward, firstReward);
        }

        actions.AddRange(bestFuture.Actions);
        float totalScore = firstReward + discount * bestFuture.TotalScore;
        return new RolloutResult(actions, firstReward, totalScore);
    }

    private static RolloutResult EvaluateHeuristicRollout(
        GameManager game,
        int firstAction,
        int depth,
        float discount)
    {
        var actions = new List<int> { firstAction };
        var simulatedCell = game.PlayerCellPosition;
        float totalScore = 0f;
        float firstScore = 0f;

        for (int step = 0; step < depth; step++)
        {
            int action = step == 0 ? firstAction : BestHeuristicActionFromCell(game, simulatedCell);
            if (step > 0)
            {
                actions.Add(action);
            }

            float stepScore = HeuristicScoreFromCell(game, simulatedCell, action);
            if (step == 0)
            {
                firstScore = stepScore;
            }

            totalScore += Mathf.Pow(discount, step) * stepScore;
            var targetCell = simulatedCell + RogueObservationBuilder.ActionToDirection(action);
            var cell = game.BoardManager.GetCellData(targetCell);
            int objectCode = game.BoardManager.GetCellObjectCode(targetCell);
            if (cell == null || !cell.Passable || objectCode == 2 || objectCode == 3)
            {
                break;
            }

            simulatedCell = targetCell;
        }

        return new RolloutResult(actions, firstScore, totalScore);
    }

    private static int BestHeuristicActionFromCell(GameManager game, Vector2Int cell)
    {
        int bestAction = 0;
        float bestScore = float.NegativeInfinity;
        for (int action = 0; action < RogueObservationBuilder.ActionCount; action++)
        {
            float score = HeuristicScoreFromCell(game, cell, action);
            if (score > bestScore)
            {
                bestScore = score;
                bestAction = action;
            }
        }

        return bestAction;
    }

    private static float HeuristicScoreFromCell(GameManager game, Vector2Int cell, int action)
    {
        var target = cell + RogueObservationBuilder.ActionToDirection(action);
        var targetData = game.BoardManager.GetCellData(target);
        if (targetData == null || !targetData.Passable)
        {
            return InvalidScore;
        }

        int currentDistance = game.DistanceToExit(cell);
        int targetDistance = game.DistanceToExit(target);
        float score = currentDistance - targetDistance;
        int objectCode = game.BoardManager.GetCellObjectCode(target);

        if (objectCode == 1)
        {
            score += 10f;
        }
        else if (objectCode == 2)
        {
            score -= 4f;
        }
        else if (objectCode == 3)
        {
            score -= 3f;
        }
        else if (objectCode == 4)
        {
            score += 0.5f;
        }

        return score;
    }

    private static string BuildActionReason(GameManager game, int action, float score, MissionActionScore missionScore)
    {
        int objectCode = GetFirstStepObjectCode(game, action);
        if (IsInvalidScore(score))
        {
            return "blocked or unsafe";
        }

        if (objectCode == 1)
        {
            return "exit is directly reachable";
        }

        if (objectCode == 4)
        {
            return "collects food while preserving the route";
        }

        if (objectCode == 2)
        {
            return "enemy blocks movement, so fighting is penalized";
        }

        if (objectCode == 3)
        {
            return "obstacle costs turns before movement";
        }

        if (missionScore != null && missionScore.isPreferredPath)
        {
            return $"shortest mission path to {missionScore.targetLabel}, estimated cost {missionScore.estimatedCost:0.0}";
        }

        return missionScore != null
            ? $"mission score estimates {missionScore.estimatedCost:0.0} cost to {missionScore.targetLabel}"
            : "mission planner predicts progress toward exit";
    }

    private static string BuildSelectionExplanation(BrainActionScore selected, MissionTarget missionTarget, bool modelLoaded)
    {
        string source = modelLoaded
            ? "mission planner plus world model rollout"
            : "mission planner fallback";
        string target = missionTarget != null ? missionTarget.label.ToUpperInvariant() : "EXIT";
        return $"Choose {selected.actionName.ToUpperInvariant()} because {source} gives the best path toward {target} ({selected.rolloutScore:0.00}). {selected.reason}.";
    }

    private static string BuildFutureSummary(List<int> actions, MissionTarget missionTarget, bool modelLoaded)
    {
        string prefix = modelLoaded ? "model-checked mission rollout" : "map mission rollout";
        string path = string.Join(" -> ", actions.Select(RogueObservationBuilder.ActionName));
        string target = missionTarget != null ? missionTarget.label : "exit";
        return $"{prefix} to {target}: {path}";
    }

    private static MissionTarget ChooseMissionTarget(GameManager game)
    {
        if (game == null || game.BoardManager == null)
        {
            return null;
        }

        var exitCell = game.BoardManager.ExitCell;
        var exitPath = FindLowestCostPath(game, game.PlayerCellPosition, exitCell);
        var bestTarget = new MissionTarget
        {
            cell = exitCell,
            label = "exit",
            isFood = false,
            objectiveCost = exitPath.cost,
            path = exitPath,
        };

        float foodPressure = game.FoodAmount <= 0
            ? 1f
            : 1f - Mathf.Clamp01(game.CurrentFoodAmount / (float)game.FoodAmount);
        float foodValue = Mathf.Lerp(4f, 12f, foodPressure);
        if (game.CurrentFoodAmount <= LowFoodThreshold)
        {
            foodValue += 4f;
        }

        for (int y = 0; y < game.BoardManager.Height; y++)
        {
            for (int x = 0; x < game.BoardManager.Width; x++)
            {
                var foodCell = new Vector2Int(x, y);
                if (game.BoardManager.GetCellObjectCode(foodCell) != 4)
                {
                    continue;
                }

                var pathToFood = FindLowestCostPath(game, game.PlayerCellPosition, foodCell);
                var pathFoodToExit = FindLowestCostPath(game, foodCell, exitCell);
                if (!pathToFood.reachable || !pathFoodToExit.reachable)
                {
                    continue;
                }

                float objectiveCost = pathToFood.cost + pathFoodToExit.cost - foodValue;
                if (objectiveCost < bestTarget.objectiveCost)
                {
                    bestTarget = new MissionTarget
                    {
                        cell = foodCell,
                        label = "food",
                        isFood = true,
                        objectiveCost = objectiveCost,
                        path = pathToFood,
                    };
                }
            }
        }

        return bestTarget;
    }

    private static MissionActionScore EvaluateMissionAction(GameManager game, int action, MissionTarget missionTarget)
    {
        float mapScore = MapHeuristicScore(game, action);
        if (IsInvalidScore(mapScore) || game == null || game.BoardManager == null || missionTarget == null)
        {
            return MissionActionScore.Invalid(missionTarget);
        }

        var direction = RogueObservationBuilder.ActionToDirection(action);
        var target = game.PlayerCellPosition + direction;
        float firstStepCost = EnterCost(game, target);
        if (firstStepCost >= UnreachablePathCost)
        {
            return MissionActionScore.Invalid(missionTarget);
        }

        var remainder = target == missionTarget.cell
            ? PathSearchResult.Reached()
            : FindLowestCostPath(game, target, missionTarget.cell);
        if (!remainder.reachable)
        {
            return MissionActionScore.Invalid(missionTarget);
        }

        float estimatedCost = firstStepCost + remainder.cost;
        bool preferredPath = missionTarget.path != null
            && missionTarget.path.actions.Count > 0
            && missionTarget.path.actions[0] == action;
        float score = MissionScoreBase - estimatedCost;
        if (preferredPath)
        {
            score += MissionPathBonus;
        }

        if (missionTarget.isFood && target == missionTarget.cell)
        {
            score += 2f;
        }

        return new MissionActionScore
        {
            Score = score,
            estimatedCost = estimatedCost,
            targetLabel = missionTarget.label,
            isPreferredPath = preferredPath,
        };
    }

    private static List<int> BuildMissionFutureActions(
        GameManager game,
        int firstAction,
        MissionTarget missionTarget,
        int depth)
    {
        var actions = new List<int> { firstAction };
        if (game == null || game.BoardManager == null || missionTarget == null || depth <= 1)
        {
            return actions;
        }

        var firstTarget = game.PlayerCellPosition + RogueObservationBuilder.ActionToDirection(firstAction);
        if (EnterCost(game, firstTarget) >= UnreachablePathCost)
        {
            return actions;
        }

        var path = firstTarget == missionTarget.cell
            ? PathSearchResult.Reached()
            : FindLowestCostPath(game, firstTarget, missionTarget.cell);
        if (path.reachable)
        {
            actions.AddRange(path.actions.Take(depth - actions.Count));
        }

        if (missionTarget.isFood && actions.Count < depth)
        {
            var exitPath = FindLowestCostPath(game, missionTarget.cell, game.BoardManager.ExitCell);
            if (exitPath.reachable)
            {
                actions.AddRange(exitPath.actions.Take(depth - actions.Count));
            }
        }

        return actions;
    }

    private static PathSearchResult FindLowestCostPath(GameManager game, Vector2Int start, Vector2Int goal)
    {
        if (game == null || game.BoardManager == null)
        {
            return PathSearchResult.Unreachable();
        }

        var board = game.BoardManager;
        if (!IsInside(board, start) || !IsInside(board, goal))
        {
            return PathSearchResult.Unreachable();
        }

        if (start == goal)
        {
            return PathSearchResult.Reached();
        }

        float[,] costs = new float[board.Width, board.Height];
        bool[,] visited = new bool[board.Width, board.Height];
        Vector2Int[,] previous = new Vector2Int[board.Width, board.Height];
        int[,] previousAction = new int[board.Width, board.Height];
        var open = new List<Vector2Int>();

        for (int y = 0; y < board.Height; y++)
        {
            for (int x = 0; x < board.Width; x++)
            {
                costs[x, y] = UnreachablePathCost;
                previousAction[x, y] = -1;
            }
        }

        costs[start.x, start.y] = 0f;
        open.Add(start);

        while (open.Count > 0)
        {
            int bestIndex = 0;
            float bestCost = costs[open[0].x, open[0].y];
            for (int i = 1; i < open.Count; i++)
            {
                float candidateCost = costs[open[i].x, open[i].y];
                if (candidateCost < bestCost)
                {
                    bestIndex = i;
                    bestCost = candidateCost;
                }
            }

            var current = open[bestIndex];
            open.RemoveAt(bestIndex);
            if (visited[current.x, current.y])
            {
                continue;
            }

            visited[current.x, current.y] = true;
            if (current == goal)
            {
                return ReconstructPath(start, goal, costs[goal.x, goal.y], previous, previousAction);
            }

            for (int action = 0; action < RogueObservationBuilder.ActionCount; action++)
            {
                var next = current + RogueObservationBuilder.ActionToDirection(action);
                if (!IsInside(board, next) || visited[next.x, next.y])
                {
                    continue;
                }

                float stepCost = EnterCost(game, next);
                if (stepCost >= UnreachablePathCost)
                {
                    continue;
                }

                float nextCost = costs[current.x, current.y] + stepCost;
                if (nextCost >= costs[next.x, next.y])
                {
                    continue;
                }

                costs[next.x, next.y] = nextCost;
                previous[next.x, next.y] = current;
                previousAction[next.x, next.y] = action;
                open.Add(next);
            }
        }

        return PathSearchResult.Unreachable();
    }

    private static PathSearchResult ReconstructPath(
        Vector2Int start,
        Vector2Int goal,
        float cost,
        Vector2Int[,] previous,
        int[,] previousAction)
    {
        var actions = new List<int>();
        var current = goal;
        while (current != start)
        {
            int action = previousAction[current.x, current.y];
            if (action < 0)
            {
                return PathSearchResult.Unreachable();
            }

            actions.Add(action);
            current = previous[current.x, current.y];
        }

        actions.Reverse();
        return new PathSearchResult
        {
            reachable = true,
            cost = cost,
            actions = actions,
        };
    }

    private static float EnterCost(GameManager game, Vector2Int cell)
    {
        var data = game.BoardManager.GetCellData(cell);
        if (data == null || !data.Passable)
        {
            return UnreachablePathCost;
        }

        int objectCode = game.BoardManager.GetCellObjectCode(cell);
        float cost;
        switch (objectCode)
        {
            case 2:
                cost = 9f;
                break;
            case 3:
                cost = 5f;
                break;
            default:
                cost = 1f;
                break;
        }

        if (objectCode != 1 && objectCode != 2 && HasAdjacentEnemy(game, cell))
        {
            cost += 2f;
        }

        return cost;
    }

    private static bool HasAdjacentEnemy(GameManager game, Vector2Int cell)
    {
        for (int action = 0; action < RogueObservationBuilder.ActionCount; action++)
        {
            var adjacent = cell + RogueObservationBuilder.ActionToDirection(action);
            if (game.BoardManager.GetCellObjectCode(adjacent) == 2)
            {
                return true;
            }
        }

        return false;
    }

    private static bool IsInside(BoardManager board, Vector2Int cell)
    {
        return cell.x >= 0 && cell.x < board.Width && cell.y >= 0 && cell.y < board.Height;
    }

    private static float FirstStepSafetyScore(GameManager game, int action)
    {
        int objectCode = GetFirstStepObjectCode(game, action);
        switch (objectCode)
        {
            case 1:
                return ExitFirstStepBonus;
            case 2:
                return EnemyFirstStepPenalty;
            case 3:
                return ObstacleFirstStepPenalty;
            case 4:
                return FoodFirstStepBonus;
            default:
                return 0f;
        }
    }

    private static int GetFirstStepObjectCode(GameManager game, int action)
    {
        if (game == null || game.BoardManager == null)
        {
            return -1;
        }

        var target = game.PlayerCellPosition + RogueObservationBuilder.ActionToDirection(action);
        return game.BoardManager.GetCellObjectCode(target);
    }

    private static bool IsInvalidScore(float score)
    {
        return score <= InvalidScore + 0.001f;
    }

    private class MissionTarget
    {
        public Vector2Int cell;
        public string label;
        public bool isFood;
        public float objectiveCost;
        public PathSearchResult path;
    }

    private class MissionActionScore
    {
        public float Score;
        public float estimatedCost;
        public string targetLabel;
        public bool isPreferredPath;

        public static MissionActionScore Invalid(MissionTarget missionTarget)
        {
            return new MissionActionScore
            {
                Score = InvalidScore,
                estimatedCost = UnreachablePathCost,
                targetLabel = missionTarget != null ? missionTarget.label : "exit",
                isPreferredPath = false,
            };
        }
    }

    private class PathSearchResult
    {
        public bool reachable;
        public float cost;
        public List<int> actions = new List<int>();

        public static PathSearchResult Reached()
        {
            return new PathSearchResult
            {
                reachable = true,
                cost = 0f,
                actions = new List<int>(),
            };
        }

        public static PathSearchResult Unreachable()
        {
            return new PathSearchResult
            {
                reachable = false,
                cost = UnreachablePathCost,
                actions = new List<int>(),
            };
        }
    }

    private class RolloutResult
    {
        public RolloutResult(List<int> actions, float firstReward, float totalScore)
        {
            Actions = actions;
            FirstReward = firstReward;
            TotalScore = totalScore;
        }

        public List<int> Actions { get; }
        public float FirstReward { get; }
        public float TotalScore { get; }
    }

    /// <summary>
    /// Summary of one <c>POST /plan_actions</c> response, reshaped for
    /// <see cref="DecideWithSidecarPlan"/>. The Unity client (see
    /// <see cref="LewmClient.PlanResponse"/>) holds the wire format with
    /// the top-k actions flattened row-major; this struct is the
    /// inflated 2D view the planner reasons about.
    ///
    /// <see cref="BestActions"/> and <c>TopKActions[0]</c> always
    /// reference the same sequence.
    /// </summary>
    public class SidecarPlanSummary
    {
        public int[] BestActions;
        public float BestScore;
        public int[][] TopKActions;
        public float[] TopKScores;
        public int Horizon;
        public int NumCandidates;

        /// <summary>
        /// Inflate the flat <see cref="LewmClient.PlanResponse"/> wire
        /// format into the 2-D shape <see cref="DecideWithSidecarPlan"/>
        /// expects. Returns <c>null</c> if the response is malformed
        /// (caller should fall back to the mission planner).
        /// </summary>
        public static SidecarPlanSummary FromPlanResponse(LewmClient.PlanResponse response)
        {
            if (response == null
                || response.best_actions == null
                || response.best_actions.Length == 0
                || response.horizon <= 0)
            {
                return null;
            }

            int topK = response.top_k;
            if (topK <= 0)
            {
                topK = response.top_k_scores != null ? response.top_k_scores.Length : 0;
            }
            int horizon = response.horizon;

            int[][] grid;
            float[] scores;
            if (response.top_k_actions_flat != null
                && response.top_k_scores != null
                && topK > 0
                && response.top_k_actions_flat.Length >= topK * horizon)
            {
                grid = new int[topK][];
                for (int k = 0; k < topK; k++)
                {
                    var row = new int[horizon];
                    for (int t = 0; t < horizon; t++)
                    {
                        row[t] = response.top_k_actions_flat[k * horizon + t];
                    }
                    grid[k] = row;
                }
                scores = new float[topK];
                for (int k = 0; k < topK; k++)
                {
                    scores[k] = k < response.top_k_scores.Length ? response.top_k_scores[k] : 0f;
                }
            }
            else
            {
                // Fall back to a one-row "top-k" containing just the best plan.
                grid = new int[][] { response.best_actions };
                scores = new float[] { response.best_score };
            }

            return new SidecarPlanSummary
            {
                BestActions = response.best_actions,
                BestScore = response.best_score,
                TopKActions = grid,
                TopKScores = scores,
                Horizon = horizon,
                NumCandidates = response.num_candidates,
            };
        }
    }
}
