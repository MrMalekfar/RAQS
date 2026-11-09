"""Windows ownership boundary for one GUI calculation and all of its descendants."""
import ctypes
from ctypes import wintypes as wt


class BasicLimits(ctypes.Structure):
    _fields_ = [('times', ctypes.c_longlong * 2), ('flags', wt.DWORD),
                ('minimum', ctypes.c_size_t), ('maximum', ctypes.c_size_t),
                ('active_limit', wt.DWORD), ('affinity', ctypes.c_size_t),
                ('priority', wt.DWORD), ('scheduling', wt.DWORD)]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [('basic', BasicLimits), ('io_counters', ctypes.c_ulonglong * 6),
                ('process_memory', ctypes.c_size_t), ('job_memory', ctypes.c_size_t),
                ('peak_process_memory', ctypes.c_size_t), ('peak_job_memory', ctypes.c_size_t)]


class Accounting(ctypes.Structure):
    _fields_ = [('times', ctypes.c_longlong * 4), ('page_faults', wt.DWORD),
                ('total', wt.DWORD), ('active', wt.DWORD), ('terminated', wt.DWORD)]


class ProcessJob:
    def __init__(self, name, *, create=False):
        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        for function, arguments, result in (
            ('CreateJobObjectW', [ctypes.c_void_p, wt.LPCWSTR], wt.HANDLE),
            ('OpenJobObjectW', [wt.DWORD, wt.BOOL, wt.LPCWSTR], wt.HANDLE),
            ('SetInformationJobObject', [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD], wt.BOOL),
            ('QueryInformationJobObject', [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD, ctypes.c_void_p], wt.BOOL),
            ('OpenProcess', [wt.DWORD, wt.BOOL, wt.DWORD], wt.HANDLE),
            ('AssignProcessToJobObject', [wt.HANDLE, wt.HANDLE], wt.BOOL),
            ('IsProcessInJob', [wt.HANDLE, wt.HANDLE, ctypes.POINTER(wt.BOOL)], wt.BOOL),
            ('TerminateJobObject', [wt.HANDLE, wt.UINT], wt.BOOL),
            ('CloseHandle', [wt.HANDLE], wt.BOOL),
        ):
            method = getattr(self.api, function); method.argtypes = arguments; method.restype = result
        self.handle = self.api.CreateJobObjectW(None, name) if create else self.api.OpenJobObjectW(0x0004 | 0x0008, False, name)
        self.check(self.handle)
        if create:
            limits = ExtendedLimits(); limits.basic.flags = 0x2000  # KILL_ON_JOB_CLOSE; no breakaway permission.
            try:
                self.check(self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
            except Exception:
                self.close(); raise

    @staticmethod
    def check(result):
        if not result:
            raise ctypes.WinError(ctypes.get_last_error())

    def assign(self, pid):
        process = self.api.OpenProcess(0x1101, False, pid)
        self.check(process)
        try:
            self.check(self.api.AssignProcessToJobObject(self.handle, process))
        finally:
            self.api.CloseHandle(process)

    def contains(self, pid):
        process = self.api.OpenProcess(0x1000, False, pid)
        self.check(process)
        try:
            member = wt.BOOL()
            self.check(self.api.IsProcessInJob(process, self.handle, ctypes.byref(member)))
            return bool(member.value)
        finally:
            self.api.CloseHandle(process)

    def active_processes(self):
        info = Accounting()
        self.check(self.api.QueryInformationJobObject(self.handle, 1, ctypes.byref(info), ctypes.sizeof(info), None))
        return info.active

    def terminate(self):
        self.check(self.api.TerminateJobObject(self.handle, 130))

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None
