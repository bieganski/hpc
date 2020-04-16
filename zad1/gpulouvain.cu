#include <iostream>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

using namespace std;

static float MIN_GAIN;
static char *FILE_PATH;
static bool VERBOSE = 0;


int parse_args (int argc, char **argv) {
    int index;
    int gflag = 0, fflag = 0;
    int c;

    opterr = 0;

    while ((c = getopt (argc, argv, "f:g:v")) != -1)
        switch (c) {
            case 'v':
                VERBOSE = 1;
                break;
            case 'f':
                fflag = 1;
                FILE_PATH = optarg;
                break;
            case 'g':
                gflag = 1;
                try {
                    MIN_GAIN = std::stof(std::string(optarg));
                } catch (std::invalid_argument) {
                    fprintf (stderr, "Min Gain must be proper float number!\n");
                    return 1;
                }
                break;
            case '?':
                if (optopt == 'f' || optopt == 'g')
                    fprintf (stderr, "Option -%c requires an argument.\n", optopt);
                else if (isprint (optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr,
                            "Unknown option character `\\x%x'.\n",
                            optopt);
                return 1;
            default:
                abort ();
            }

    if (!fflag) {
        fprintf (stderr, "Filename option -f is required!");
        return 1;
    } else if(!gflag) {
        fprintf (stderr, "Min gain option -g is required!");
        return 1;
    }

    return 0;
}

int main(int argc, char **argv) {
    if (parse_args(argc, argv)) {
        exit(1);
    }

    cout << MIN_GAIN << ", " << FILE_PATH << ", " << VERBOSE;

    return 0;
}