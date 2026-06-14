/* Use-After-Free (CWE-416) - access memory after deallocation */
void handle_session(int session_id) {
    char *data = NULL;
    char *backup = NULL;
    int result = 0;

    data = (char *)malloc(256);
    if (data == NULL) {
        return;
    }
    backup = (char *)malloc(128);
    if (backup == NULL) {
        free(data);
        return;
    }

    data[0] = 'A';
    data[1] = 'B';

    /* Copy data before freeing */
    memcpy(backup, data, 128);

    /* Free the data buffer */
    free(data);

    /* USE AFTER FREE: reading freed memory */
    result = data[0];

    /* USE AFTER FREE: writing to freed memory */
    data[1] = 'C';

    if (result > 0) {
        backup[0] = data[0];
    }

    /* DOUBLE FREE */
    free(data);
    free(backup);
}
