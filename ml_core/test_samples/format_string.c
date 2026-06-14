/* Format String Vulnerability (CWE-134) - user input as format string */
void log_user_activity(char *username, char *action) {
    char log_buf[512];
    char temp[256];
    int log_level = 1;

    /* VULNERABLE: username directly as format string */
    printf(username);

    /* VULNERABLE: action used as format string in fprintf */
    fprintf(stderr, action);

    /* VULNERABLE: user data as format specifier in sprintf */
    sprintf(log_buf, username);

    /* VULNERABLE: syslog with user controlled format */
    syslog(log_level, action);

    /* Safe comparison for contrast */
    if (log_level > 0) {
        log_level = log_level + 1;
    }
}
