import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from easyAI import AI_Player, Negamax, TwoPlayerGame


TOTAL_GAMES = 1024
NUM_THREADS = 4
BASE_SEED = 42

EXPERIMENTS = [
    # 1. Porównanie Negamax (Alfa-Beta) vs Pure Negamax (Bez odcięć) - Deterministyczne
    {"name": "D_Negamax_AB_d4", "algo": "negamax", "depth": 4, "prob": 0.0},
    {"name": "D_PureNegamax_d4", "algo": "pure", "depth": 4, "prob": 0.0},

    # 2. Porównanie głębokości na wariancie probabilistycznym
    {"name": "P_Negamax_AB_d3", "algo": "negamax", "depth": 3, "prob": 0.2},
    {"name": "P_Negamax_AB_d5", "algo": "negamax", "depth": 5, "prob": 0.2},

    # 3. Expectiminimax - algorytm dedykowany do losowości
    {"name": "P_Expectiminimax_d3", "algo": "expecti", "depth": 3, "prob": 0.2},
]



class PureNegamax:
    """Wersja Negamax BEZ odcięcia Alfa-Beta (przeszukuje wszystko)"""

    def __init__(self, depth, scoring=None):
        self.depth = depth

    def __call__(self, game):
        return self.solve(game, self.depth)

    def solve(self, game, depth):
        if depth == 0 or game.is_over():
            return game.scoring()

        possible_moves = game.possible_moves()
        best_value = -float('inf')
        best_move = possible_moves[0]

        for move in possible_moves:
            game.make_move(move)
            game.switch_player()
            value = -self.solve(game, depth - 1)
            game.switch_player()
            game.unmake_move(move)

            if value > best_value:
                best_value = value
                best_move = move

        if depth == self.depth:
            return best_move
        return best_value


class Expectiminimax:
    """Algorytm Expectiminimax uwzględniający szansę na niepowodzenie ruchu"""

    def __init__(self, depth, failure_prob=0.2):
        self.depth = depth
        self.failure_prob = failure_prob

    def __call__(self, game):
        possible_moves = game.possible_moves()
        best_move = None
        best_value = -float('inf')

        for move in possible_moves:
            game.make_move_forced(move, success=True)
            game.switch_player()
            val_success = -self.solve(game, self.depth - 1)
            game.switch_player()
            game.unmake_move_forced(success=True)

            game.make_move_forced(move, success=False)
            game.switch_player()
            val_fail = -self.solve(game, self.depth - 1)
            game.switch_player()
            game.unmake_move_forced(success=False)

            expected_value = (1 - self.failure_prob) * val_success + (self.failure_prob * val_fail)

            if expected_value > best_value:
                best_value = expected_value
                best_move = move
        return best_move

    def solve(self, game, depth):
        if depth == 0 or game.is_over():
            return game.scoring()

        res = []
        for move in game.possible_moves():
            game.make_move_forced(move, success=True)
            game.switch_player()
            v = -self.solve(game, depth - 1)
            game.switch_player()
            game.unmake_move_forced(success=True)
            res.append(v)

        return max(res) if res else game.scoring()


class TicTacDoh(TwoPlayerGame):
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
        """Standardowy ruch z losowością"""
        success = self.rng.random() > self.failure_prob
        self.make_move_forced(move, success)

    def make_move_forced(self, move, success):
        """Pomocnicze do Expectiminimaxa i Unmake"""
        if success:
            self.board[int(move) - 1] = self.current_player
            self.move_executed_stack.append(True)
        else:
            self.move_executed_stack.append(False)

    def unmake_move(self, move):
        executed = self.move_executed_stack.pop()
        if executed:
            self.board[int(move) - 1] = 0

    def unmake_move_forced(self, success):
        if success:
            self.board[
                self.move_executed_stack.index(True) if True in self.move_executed_stack else 0] = 0  # uproszczenie
            self.move_executed_stack.pop() if self.move_executed_stack else None
        else:
            self.move_executed_stack.pop() if self.move_executed_stack else None

    def unmake_move(self, move):
        if not self.move_executed_stack: return
        executed = self.move_executed_stack.pop()
        if executed:
            self.board[int(move) - 1] = 0

    def lose(self):
        return any([all([(self.board[c - 1] == self.opponent_index) for c in line])
                    for line in
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 4, 7], [2, 5, 8], [3, 6, 9], [1, 5, 9], [3, 5, 7]]])

    def is_over(self):
        return (self.possible_moves() == []) or self.lose()

    def scoring(self):
        return -100 if self.lose() else 0


def get_ai(algo_type, depth, prob):
    if algo_type == "negamax": return Negamax(depth)
    if algo_type == "pure": return PureNegamax(depth)
    if algo_type == "expecti": return Expectiminimax(depth, prob)


def run_game(exp1, exp2, seed):
    rng = random.Random(seed)
    ai1 = get_ai(exp1['algo'], exp1['depth'], exp1['prob'])
    ai2 = get_ai(exp2['algo'], exp2['depth'], exp2['prob'])

    game = TicTacDoh([AI_Player(ai1), AI_Player(ai2)], failure_prob=exp1['prob'], rng=rng)

    move_times = []

    while not game.is_over():
        player_ai = ai1 if game.current_player == 1 else ai2

        start = time.perf_counter()
        move = player_ai(game)
        end = time.perf_counter()

        move_times.append((game.current_player, end - start))
        game.make_move(move)
        game.switch_player()

    winner = 0
    if game.lose():
        winner = 3 - game.current_player

    return winner, move_times


def run_experiment(exp):
    print(f"Uruchamianie eksperymentu: {exp['name']}...")
    stats = {"wins": 0, "losses": 0, "draws": 0, "total_moves": 0, "total_time": 0.0}

    opponent_exp = {"algo": "negamax", "depth": 3, "prob": exp['prob']}

    for i in range(TOTAL_GAMES):
        swapped = i % 2 != 0
        if not swapped:
            winner, times = run_game(exp, opponent_exp, BASE_SEED + i)
        else:
            winner, times = run_game(opponent_exp, exp, BASE_SEED + i)
            if winner != 0: winner = 3 - winner

        if winner == 1:
            stats["wins"] += 1
        elif winner == 2:
            stats["losses"] += 1
        else:
            stats["draws"] += 1

        target_p = 1 if not swapped else 2
        for p, t in times:
            if p == target_p:
                stats["total_time"] += t
                stats["total_moves"] += 1

    avg_time = (stats["total_time"] / stats["total_moves"] * 1000) if stats["total_moves"] > 0 else 0

    return {
        "name": exp['name'],
        "wld": f"{stats['wins']}/{stats['losses']}/{stats['draws']}",
        "wr": (stats['wins'] / TOTAL_GAMES) * 100,
        "time": avg_time
    }


def main():
    results = [run_experiment(e) for e in EXPERIMENTS]
    print("\n" + "=" * 80)
    print(f"{'Eksperyment':<25} | {'W/L/D':<15} | {'WinRate':<10} | {'Czas/Ruch (ms)':<15}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<25} | {r['wld']:<15} | {r['wr']:>8.1f}% | {r['time']:>12.4f} ms")
    print("=" * 80)


if __name__ == "__main__":
    main()