/* Safe Code - properly bounded operations, no vulnerabilities */
int process_records(int count) {
    int total = 0;
    int average = 0;
    int i;
    int records[100];

    if (count > 100) {
        count = 100;
    }
    if (count < 0) {
        count = 0;
    }

    for (i = 0; i < count; i++) {
        records[i] = i * 2 + 1;
        total = total + records[i];
    }

    if (count > 0) {
        average = total / count;
    }

    return average;
}
