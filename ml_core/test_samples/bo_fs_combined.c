/* Combined: Buffer Overflow (CWE-121) + Format String (CWE-134) */
void process_user_request(char *user_input, int input_len) {
    char local_buf[64];
    char log_buf[256];
    int status = 0;

    /* BUFFER OVERFLOW: unbounded copy into fixed buffer */
    strcpy(local_buf, user_input);

    /* BUFFER OVERFLOW: copy with user-controlled size */
    memcpy(log_buf, user_input, input_len);

    /* FORMAT STRING: user input directly as format string */
    printf(user_input);

    /* FORMAT STRING: user data in fprintf format position */
    fprintf(stderr, local_buf);

    /* BUFFER OVERFLOW: unsafe concatenation */
    strcat(local_buf, log_buf);

    /* FORMAT STRING: sprintf with user-controlled format */
    sprintf(log_buf, user_input);

    if (status == 0) {
        status = 1;
    }
}
