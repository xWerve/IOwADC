import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from easyAI import AI_Player, Negamax, NonRecursiveNegamax, TwoPlayerGame


# =========================
# HARD-CODED SETTINGS
# =========================
TOTAL_GAMES = 1024
NUM_THREADS = 8
BASE_SEED = 42
VERBOSE_PLAY = False
PROGRESS_EVERY = 0

EXPERIMENTS = [
    {
        "name": "deterministic_negamax_d3_vs_d5",
        "algo": "negamax",
        "depth1": 3,
        "depth2": 5,
        "failure_prob": 0.0,
    },
    {
        "name": "probabilistic_negamax_d3_vs_d5",
        "algo": "negamax",
        "depth1": 3,
        "depth2": 5,
        "failure_prob": 0.2,
    },
]


class TicTacDoh(TwoPlayerGame):
    """The board positions are numbered as follows:
    1 2 3
    4 5 6
    7 8 9
    """

    def __init__(self, players, failure_prob=0.2, rng=None):
        self.players = players
        self.board = [0 for _ in range(9)]
        self.current_player = 1
        self.failure_prob = failure_prob
        self.rng = rng if rng is not None else random.Random()
        self.move_executed_stack = []

    def possible_moves(self):
        return [i + 1 for i, e in enumerate(self.board) if e == 0]

    def make_move(self, move):
        if self.rng.random() > self.failure_prob:
            self.board[int(move) - 1] = self.current_player
            self.move_executed_stack.append(True)
        else:
            self.move_executed_stack.append(False)

    def unmake_move(self, move):
        executed = self.move_executed_stack.pop()
        if executed:
            self.board[int(move) - 1] = 0

    def lose(self):
        return any(
            [
                all([(self.board[c - 1] == self.opponent_index) for c in line])
                for line in [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [1, 4, 7],
                    [2, 5, 8],
                    [3, 6, 9],
                    [1, 5, 9],
                    [3, 5, 7],
                ]
            ]
        )

    def is_over(self):
        return (self.possible_moves() == []) or self.lose()

    def show(self):
        print(
            "\n"
            + "\n".join(
                [
                    " ".join([[".", "O", "X"][self.board[3 * j + i]] for i in range(3)])
                    for j in range(3)
                ]
            )
        )

    def scoring(self):
        return -100 if self.lose() else 0

    def ttentry(self):
        return (tuple(self.board), self.current_player, tuple(self.move_executed_stack))

    def ttrestore(self, entry):
        board, current_player, move_stack = entry
        self.board = list(board)
        self.current_player = current_player
        self.move_executed_stack = list(move_stack)


def create_ai(algo_name, depth):
    if algo_name == "negamax":
        return Negamax(depth)
    if algo_name == "nonrecursive":
        return NonRecursiveNegamax(depth)
    raise ValueError(f"Unsupported algo: {algo_name}")


def split_games(total_games, num_threads):
    base = total_games // num_threads
    rest = total_games % num_threads
    return [base + (1 if i < rest else 0) for i in range(num_threads)]


def run_batch(batch_id, n_games, exp, seed):
    results = {"p1_wins": 0, "p2_wins": 0, "draws": 0}
    half = n_games // 2
    rng = random.Random(seed)

    for i in range(n_games):
        if i < half:
            ai_first = create_ai(exp["algo"], exp["depth1"])
            ai_second = create_ai(exp["algo"], exp["depth2"])
            swapped = False
        else:
            ai_first = create_ai(exp["algo"], exp["depth2"])
            ai_second = create_ai(exp["algo"], exp["depth1"])
            swapped = True

        game = TicTacDoh(
            [AI_Player(ai_first), AI_Player(ai_second)],
            failure_prob=exp["failure_prob"],
            rng=rng,
        )
        game.play(verbose=VERBOSE_PLAY)

        if game.lose():
            winner = 3 - game.current_player
            if swapped:
                winner = 3 - winner
            if winner == 1:
                results["p1_wins"] += 1
            else:
                results["p2_wins"] += 1
        else:
            results["draws"] += 1

        if PROGRESS_EVERY > 0 and (i + 1) % PROGRESS_EVERY == 0:
            print(f"  Batch {batch_id}: {i + 1}/{n_games}")

    return results


def run_experiment(exp, index):
    if not (0.0 <= exp["failure_prob"] <= 1.0):
        raise ValueError(f"{exp['name']}: failure_prob must be in [0,1]")

    print("\n" + "=" * 70)
    print(
        f"Experiment {index + 1}: {exp['name']} | "
        f"algo={exp['algo']} | depths=({exp['depth1']},{exp['depth2']}) | "
        f"failure_prob={exp['failure_prob']}"
    )
    print("=" * 70)

    totals = {"p1_wins": 0, "p2_wins": 0, "draws": 0}
    games_distribution = split_games(TOTAL_GAMES, NUM_THREADS)
    active_batches = [(i, n) for i, n in enumerate(games_distribution) if n > 0]

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {
            executor.submit(run_batch, batch_id, batch_games, exp, BASE_SEED + 1000 * index + batch_id): batch_id
            for batch_id, batch_games in active_batches
        }

        for future in as_completed(futures):
            result = future.result()
            for key in totals:
                totals[key] += result[key]

    total = sum(totals.values())
    print(f"RESULTS AFTER {total} GAMES")
    print(f"Player 1 wins: {totals['p1_wins']:4d} ({totals['p1_wins']/total*100:.1f}%)")
    print(f"Player 2 wins: {totals['p2_wins']:4d} ({totals['p2_wins']/total*100:.1f}%)")
    print(f"Draws:         {totals['draws']:4d} ({totals['draws']/total*100:.1f}%)")
    print(f"Player 1 edge: {(totals['p1_wins'] - totals['p2_wins']):+d} games")


def main():
    if TOTAL_GAMES <= 0:
        raise ValueError("TOTAL_GAMES must be > 0")
    if NUM_THREADS <= 0:
        raise ValueError("NUM_THREADS must be > 0")

    print(
        f"Starting benchmarks: TOTAL_GAMES={TOTAL_GAMES}, "
        f"NUM_THREADS={NUM_THREADS}, VERBOSE_PLAY={VERBOSE_PLAY}"
    )

    for i, exp in enumerate(EXPERIMENTS):
        run_experiment(exp, i)


if __name__ == "__main__":
    main()
