#include <platform.h>

void audio_gen(chanend c[8]);
void worker_tile(chanend c[8]);

int main(){
    chan c[8];
    par {
        on tile[0] : audio_gen(c);
        on tile[1] : worker_tile(c);
    }
    return 0;
}