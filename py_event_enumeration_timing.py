import timeit
import numpy as np


def _generate_tau_i_events(t_index: int, mt_index: int, assignments: list[int]) -> list[list[int]]:
    M = assignments.copy()
    M.pop(t_index)
    u = [] if mt_index == 0 else [mt_index]
    if 0 in u:
        u.remove(0)

    events = []
    _enumerate_events(M, events, u, [])

    for e in events:
        e.insert(t_index, mt_index)

    return events


def _enumerate_events(M, E, u, v, d=0) -> None:
    if d == len(M):
        E.append(v)
        return
    
    for i in M[d]:
        if i not in u:
            vnew = v.copy()
            vnew.append(i)
            unew = u.copy()
            # feasible events can have multiple tracks with no measurement assigned
            if i != 0:
                unew.append(i)
            _enumerate_events(M, E, unew, vnew, d+1)

def main():
    t_index = 3
    mt_index = 5
    M = [[0, 1, 2, 4, 5, 7],
            [0, 2, 4, 5], 
            [0, 1, 3], 
            [0, 3, 4, 1], 
            [0, 2, 3, 6], 
            [0, 1, 8], 
            [0, 2, 4, 6], 
            [0, 1, 2, 3, 4, 5, 6, 7, 8], 
            [0, 4, 6, 8], 
            [0, 7, 8, 9], 
            [0, 5], 
            [0, 1, 3, 5, 9, 10], 
            [0, 5, 6, 7, 10], 
            [0, 1, 2, 8, 9, 10], 
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [0, 3, 5, 7, 8],
        ]

    def fn():
        events = _generate_tau_i_events(t_index, mt_index, M)
        print("Completed a cycle")

    N = 50
    elapsed_time = timeit.timeit(stmt=fn , number=N)
    print(f'elapsed time: {elapsed_time}')


if __name__ == "__main__":
    main()


