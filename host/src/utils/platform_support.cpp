// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "utils.hpp"

#if defined(__linux__)
#include <dlfcn.h>          // dlopen/dlsym/dlerror/dlclose
#include <unistd.h>         // readlink
#include <sys/ioctl.h>      // ioctl
#include <linux/limits.h>   // PATH_MAX

#elif defined(__APPLE__)
#include <dlfcn.h>          // dlopen/dlsym/dlerror/dlclose
#include <unistd.h>         // readlink
#include <mach-o/dyld.h>    // _NSGetExecutablePath
#include <sys/ioctl.h>      // ioctl

#elif defined(_WIN32)
#include <Windows.h>        // GetModuleFileNameA
#include <errhandlingapi.h> // GetLastError
#include <direct.h>         // _getcwd
#define getcwd _getcwd
#define PATH_MAX 1000
#else
#error "Unknown Operating System"
#endif

using namespace std;

string convert_to_abs_path(const string rel_path)
{

    string dir_path_str;
    char path[PATH_MAX];
    if (getcwd(path, sizeof(path)) == NULL) {
        cerr << "Could not find current working directory path " << endl;
        exit(HOST_APP_ERROR);
    }
    dir_path_str = path;
#if defined(_WIN32)
    dir_path_str += '\\';
#else
    dir_path_str += "/";
#endif
    string file_path_str = dir_path_str + rel_path;

    return file_path_str;
}

string get_dynamic_lib_path(const string lib_name)
{

#if defined(__linux__)
    string full_lib_name = "lib" + lib_name;
    char * dir_path;
    char path[PATH_MAX];
    full_lib_name = '/' + full_lib_name;
    ssize_t count = readlink("/proc/self/exe", path, PATH_MAX);
    if (count == -1)
    {
        cerr << "Could not read /proc/self/exe into " << PATH_MAX << " string" << endl;
        exit(HOST_APP_ERROR);
    }
    full_lib_name += ".so";
    path[count] = '\0'; // readlink doesn't always add NULL for some reason
#elif defined(__APPLE__)
    string full_lib_name = "lib" + lib_name;
    char * dir_path;
    char path[PATH_MAX];
    full_lib_name = '/' + full_lib_name;
    uint32_t size = PATH_MAX;
    if (_NSGetExecutablePath(path, &size) != 0)
    {
        cerr << "Could not get binary path into " << PATH_MAX << " string" << endl;
        exit(HOST_APP_ERROR);
    }
    full_lib_name += ".dylib";
#elif defined(_WIN32)
    string full_lib_name = lib_name;
    full_lib_name = '\\' + full_lib_name;
    char path[MAX_PATH];
    if(0 == GetModuleFileNameA(GetModuleHandle(NULL), path, MAX_PATH))
    {
        cerr << "Could not get binary path into " << MAX_PATH << " string" << endl;
        exit(HOST_APP_ERROR);
    }
    full_lib_name += ".dll";
#endif // linux vs apple vs windows
    string full_path = path;
    size_t found = full_path.find_last_of("/\\"); // works for both unix and windows
    string dir_path_str = full_path.substr(0, found);
    string lib_path_str = dir_path_str + full_lib_name;

    return lib_path_str;
}

dl_handle_t get_dynamic_lib(const string lib_path)
{
#if (defined(__linux__) || defined(__APPLE__))
    static_cast<void>(dlerror()); // clear errors
    dl_handle_t handle = dlopen(lib_path.c_str(), RTLD_NOW);
#elif defined(_WIN32)
    dl_handle_t handle = LoadLibrary(lib_path.c_str());
#else
#error "Unknown Operating System"
#endif // unix vs windows
    if(handle == NULL)
    {
#if (defined(__linux__) || defined(__APPLE__))
        cerr << dlerror() << endl;
#elif defined(_WIN32)
        cerr << "Could not load " << lib_path << ", got " << GetLastError() << " error code" << endl;
#else
#error "Unknown Operating System"
#endif // unix vs windows
        exit(HOST_APP_ERROR);
    }
    return handle;
}

template<typename T>
T get_function(dl_handle_t handle, const string symbol)
{
#if (defined(__linux__) || defined(__APPLE__))
    static_cast<void>(dlerror()); // clear errors
    T func = reinterpret_cast<T>(dlsym(handle, symbol.c_str()));
#elif defined(_WIN32)
    T func = reinterpret_cast<T>(GetProcAddress(handle, symbol.c_str()));
#else
#error "Unsupported operating system"
#endif // unix vs windows
    if(func == NULL)
    {
#if (defined(__linux__) || defined(__APPLE__))
        cerr << dlerror() << endl;
#elif defined(_WIN32)
        cerr << "Could not load " << symbol << " function, got " << GetLastError() << " error code" << endl;
#else
#error "Unknown Operating System"
#endif // unix vs windows
        exit(HOST_APP_ERROR);
    }
    return func;
}

num_cmd_fptr get_num_cmd_fptr(dl_handle_t handle)
{
    return get_function<num_cmd_fptr>(handle, "get_num_commands");
}

cmd_index_fptr get_cmd_index_fptr(dl_handle_t handle)
{
    return get_function<cmd_index_fptr>(handle, "get_cmd_index");
}

cmd_name_fptr get_cmd_name_fptr(dl_handle_t handle)
{
    return get_function<cmd_name_fptr>(handle, "get_cmd_name");
}

cmd_id_info_fptr get_cmd_id_info_fptr(dl_handle_t handle)
{
    return get_function<cmd_id_info_fptr>(handle, "get_cmd_id_info");
}

cmd_val_info_fptr get_cmd_val_info_fptr(dl_handle_t handle)
{
    return get_function<cmd_val_info_fptr>(handle, "get_cmd_val_info");
}

cmd_info_fptr get_cmd_info_fptr(dl_handle_t handle)
{
    return get_function<cmd_info_fptr>(handle, "get_cmd_info");
}

device_fptr get_device_fptr(dl_handle_t handle)
{
    return get_function<device_fptr>(handle, "make_Dev");
}

device_info_fptr get_device_info_fptr(dl_handle_t handle, const string symbol)
{
    return get_function<device_info_fptr>(handle, symbol);
}

size_t get_term_width()
{
#if (defined(__linux__) || defined(__APPLE__))
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    // running on jenkins returns 0
    if(w.ws_col == 0){return 120;}
    else {return w.ws_col;}
#elif defined(_WIN32)
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
    return csbi.srWindow.Right - csbi.srWindow.Left + 1;
#else
#error "Unsupported operating system"
#endif
}
