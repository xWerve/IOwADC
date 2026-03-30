# Imports
import sys
import os
import time

# AIPython Import
sys.path.append(os.path.join(os.path.dirname(__file__), "aipython"))

from stripsProblem import STRIPS_domain, Planning_problem, Strips
from stripsForwardPlanner import Forward_STRIPS
from searchGeneric import AStarSearcher


# Actions
def move(truck, x, y):
    return Strips(
        f'move({truck},{x}->{y})',
        {f'at({truck},{x})': True},
        {f'at({truck},{y})': True, f'at({truck},{x})': False}
    )

def load(truck, package, city):
    return Strips(
        f'load({package},{truck},{city})',
        {f'at({truck},{city})': True, f'at({package},{city})': True},
        {f'in({package},{truck})': True, f'at({package},{city})': False}
    )

def unload(truck, package, city):
    return Strips(
        f'unload({package},{truck},{city})',
        {f'at({truck},{city})': True, f'in({package},{truck})': True},
        {f'at({package},{city})': True, f'in({package},{truck})': False}
    )


# Generating domens
def create_domain(cities, trucks, packages):
    actions = []
    for t in trucks:
        for c1 in cities:
            for c2 in cities:
                if c1 != c2:
                    actions.append(move(t, c1, c2))
    for t in trucks:
        for p in packages:
            for c in cities:
                actions.append(load(t, p, c))
                actions.append(unload(t, p, c))
    return STRIPS_domain("logistics", actions)


# Full state
def create_full_state(cities, trucks, packages, true_facts):
    state = {}
    for t in trucks:
        for c in cities:
            state[f'at({t},{c})'] = False
    for p in packages:
        for c in cities:
            state[f'at({p},{c})'] = False
        for t in trucks:
            state[f'in({p},{t})'] = False
    for fact in true_facts:
        state[fact] = True
    return state


# Run problem
def run_problem(name, problem):
    print(f"\n===== {name} =====")
    start = time.time()

    planner = Forward_STRIPS(problem)
    planner.heuristic = lambda state: sum(1 for prop, val in problem.goal.items()
                                          if state.assignment.get(prop, False) != val)

    searcher = AStarSearcher(planner)
    searcher.max_display_level = 0

    result = searcher.search()
    end = time.time()

    if result:
        print(f"Sukces!")
        path = result
        steps = []
        while path is not None and path.arc is not None:
            steps.append(path.arc.action)
            path = path.initial
        print(f"Liczba kroków: {len(steps)}")
        for step in reversed(steps):
            print(f"  - {step}")
    else:
        print("Brak planu")
    print(f"Czas: {end - start:.4f}s")


# Subgoals
def solve_with_subgoals(name, domain, initial, subgoals):
    print(f"\n===== {name} (METODA PODCELI) =====")
    start_total = time.time()
    current_state = initial.copy()
    full_plan = []

    for i, g in enumerate(subgoals):
        prob = Planning_problem(domain, current_state, g)
        planner = Forward_STRIPS(prob)
        planner.heuristic = lambda state: sum(1 for prop, val in g.items()
                                              if state.assignment.get(prop, False) != val)
        searcher = AStarSearcher(planner)
        searcher.max_display_level = 0
        result = searcher.search()

        if result:
            path = result
            sub_steps = []
            while path is not None and path.arc is not None:
                sub_steps.append(path.arc.action)
                path = path.initial
            full_plan.extend(reversed(sub_steps))
            current_state = result.end().assignment
        else:
            print(f"Błąd w podcelu {i + 1}")
            return

    end_total = time.time()
    print(f"Sukces!")
    print(f"Łączna liczba kroków: {len(full_plan)}")
    for step in full_plan:
        print(f"  - {step}")
    print(f"Całkowity czas: {end_total - start_total:.4f}s")


# Main
if __name__ == "__main__":
    # Problem 1
    cities1 = ['A', 'B', 'C']
    trucks1 = ['T1']
    packages1 = ['P1', 'P2', 'P3']
    domain1 = create_domain(cities1, trucks1, packages1)
    true_facts1 = {'at(T1,A)', 'at(P1,A)', 'at(P2,B)', 'at(P3,C)'}
    initial1 = create_full_state(cities1, trucks1, packages1, true_facts1)
    goal1 = {'at(P1,C)': True, 'at(P2,A)': True, 'at(P3,B)': True}
    run_problem("PROBLEM 1", Planning_problem(domain1, initial1, goal1))

    # Problem 2
    cities2 = ['A', 'B', 'C', 'D']
    trucks2 = ['T1']
    packages2 = ['P1', 'P2', 'P3', 'P4', 'P5']
    domain2 = create_domain(cities2, trucks2, packages2)
    true_facts2 = {'at(T1,A)', 'at(P1,A)', 'at(P2,A)', 'at(P3,B)', 'at(P4,C)', 'at(P5,D)'}
    initial2 = create_full_state(cities2, trucks2, packages2, true_facts2)
    subgoals2 = [
        {'at(P1,D)': True},
        {'at(P2,C)': True},
        {'at(P3,A)': True},
        {'at(P4,B)': True},
        {'at(P5,A)': True}
    ]
    solve_with_subgoals("PROBLEM 2", domain2, initial2, subgoals2)

    # Problem 3
    cities3 = ['A', 'B', 'C', 'D', 'E']
    trucks3 = ['T1']
    packages3 = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
    domain3 = create_domain(cities3, trucks3, packages3)
    true_facts3 = {'at(T1,A)', 'at(P1,A)', 'at(P2,B)', 'at(P3,C)', 'at(P4,D)', 'at(P5,E)', 'at(P6,A)'}
    initial3 = create_full_state(cities3, trucks3, packages3, true_facts3)
    subgoals3 = [
        {'at(P1,E)': True},
        {'at(P2,D)': True},
        {'at(P3,A)': True},
        {'at(P4,B)': True},
        {'at(P5,C)': True},
        {'at(P6,D)': True}
    ]
    solve_with_subgoals("PROBLEM 3", domain3, initial3, subgoals3)

    # Problem 4
    cities4 = ['A', 'B', 'C', 'D', 'E', 'F']
    trucks4 = ['T1']
    packages4 = [f'P{i}' for i in range(1, 9)]
    domain4 = create_domain(cities4, trucks4, packages4)
    true_facts4 = {'at(T1,A)'} | {f'at(P{i},A)' for i in range(1, 9)}
    initial4 = create_full_state(cities4, trucks4, packages4, true_facts4)
    subgoals4 = [{f'at(P{i},F)': True} for i in range(1, 9)]
    solve_with_subgoals("PROBLEM 4", domain4, initial4, subgoals4)