

from ast import List


def __enumerate_events(mts_matrix, event_list, used_mts: List(int),
                       event: List(int), skip: int, depth: int = 0) -> None:
    if depth is skip:
        return __enumerate_events(mts_matrix, event_list, used_mts, event, skip, depth+1)

    if depth >= len(mts_matrix):
        # Try pre-allocating the array v to be a zeros array and then set value
        # at index d
        event_list.append(event)
        return

    # print(f'depth: {depth}, mts_matrix size: {len(mts_matrix)}')
    for i in mts_matrix[depth]:
        # loop over i in M[d] \ u
        if i not in used_mts:
            vnew = event.copy()
            vnew[depth] = i
            unew = used_mts.copy()
            # feasible events can have multiple tracks with no measurement assigned
            if i != 0:
                unew.append(i)
            __enumerate_events(mts_matrix, event_list,
                               unew, vnew, skip, depth+1)


def main():
    t_index = 3
    mt_index = 5
    M = [[0, 1, 2, 4, 5, 7], [0, 2, 4, 5], [0, 1, 3], [0, 3, 4, 1], [0, 2, 3, 6], [0, 1, 8], [0, 2, 4, 6], [0, 1, 2, 3, 4, 5, 6, 7, 8], [
        0, 4, 6, 8], [0, 7, 8, 9], [0, 5], [0, 1, 3, 5, 9, 10], [0, 5, 6, 7, 10], [0, 1, 2, 8, 9, 10], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    # M.pop(t_index)
    u = [] if mt_index == 0 else [mt_index]
    if 0 in u:
        u.remove(0)

    events = []
    __enumerate_events(M, events, u, [mt_index]*len(M), t_index)
    print(len(events))

    # for e in events:
    #     e.insert(t_index, mt_index)


if __name__ == '__main__':
    main()
