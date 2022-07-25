fn main() {
    let tau = 3;
    let i = 5;

    // let assignments = vec![vec![0,1,2], vec![0,2,3], vec![0,1,3]];
    let assignments = vec![
        vec![0, 1, 2, 4, 5, 7],
        vec![0, 2, 4, 5],
        vec![0, 1, 3],
        vec![0, 3, 4, 1],
        vec![0, 2, 3, 6],
        vec![0, 1, 8],
        vec![0, 2, 4, 6],
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8],
        vec![0, 4, 6, 8],
        vec![0, 7, 8, 9],
        vec![0, 5],
        vec![0, 1, 3, 5, 9, 10],
        vec![0, 5, 6, 7, 10],
        vec![0, 1, 2, 8, 9, 10],
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        vec![0, 3, 5, 7, 8],
    ];
    for n in 0..3 {
        let _ = generate_tau_i_events(tau, i, &assignments);
        // println!("{:?}", events);
        // println!("number of events: {}", events.len());
        println!("{} cycles completed", n);
    }
}

fn generate_tau_i_events(tau: usize, i: u16, assignments: &Vec<Vec<u16>>) -> Vec<Vec<u16>> {
    let mut tau_assignments = assignments.clone();
    tau_assignments.remove(tau);
    let used = if i == 0 { Vec::new() } else { vec![i] };

    let mut events: Vec<Vec<u16>> = Vec::new();
    enumerate_events(&tau_assignments, &mut events, used, Vec::new(), 0);

    for event in &mut events {
        event.insert(tau, i);
    }

    events
}

fn enumerate_events(
    assignments: &Vec<Vec<u16>>,
    entries: &mut Vec<Vec<u16>>,
    used_mt_indices: Vec<u16>,
    event: Vec<u16>,
    depth: usize,
) {
    if depth == assignments.len() {
        entries.push(event);
        return;
    }

    for i in &assignments[depth] {
        if !used_mt_indices.contains(i) {
            let mut new_event = event.clone();
            new_event.push(*i);
            let mut new_used_mt_indices = used_mt_indices.clone();

            if *i != 0 {
                new_used_mt_indices.push(*i);
            }
            enumerate_events(
                assignments,
                entries,
                new_used_mt_indices,
                new_event,
                depth + 1,
            )
        }
    }
}
