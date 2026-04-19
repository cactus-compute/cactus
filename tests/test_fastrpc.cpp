#include <cstdio>
#include <cstdlib>
#include <windows.h>
#include <wintrust.h>
#include <softpub.h>
#include <wincrypt.h>
#include "../cactus/npu/fastrpc_drv.h"
#include "../cactus/npu/htp_protocol.h"

static const wchar_t* CAT_SEARCH_PATHS[] = {
    L"C:\\Windows\\System32\\DriverStore\\FileRepository\\qcnspmcdm8380.inf_arm64_e663a92a933cab52\\libggml-htp.cat",
    L"C:\\Users\\justi\\GitRepos\\qualcomm-npu\\third_party\\llama.cpp\\pkg-snapdragon\\lib\\libggml-htp.cat",
    nullptr
};

static const wchar_t* SKEL_PATH =
    L"C:\\Windows\\System32\\unknown\\libggml-htp-v73.so";

static void print_verify_result(LONG r) {
    printf("0x%08lx", (unsigned long)r);
    if      (r == 0)                    printf(" (TRUSTED)");
    else if (r == (LONG)0x800B0100)     printf(" (TRUST_E_NOSIGNATURE - not signed)");
    else if (r == (LONG)0x800B0109)     printf(" (CERT_E_UNTRUSTEDROOT - self-signed cert rejected)");
    else if (r == (LONG)0x800B0111)     printf(" (TRUST_E_EXPLICIT_DISTRUST)");
    else if (r == (LONG)0x800B0004)     printf(" (TRUST_E_SUBJECT_NOT_TRUSTED)");
    else if (r == (LONG)0x80092026)     printf(" (CRYPT_E_SECURITY_SETTINGS - policy blocks)");
    else if (r == (LONG)0x800B010E)     printf(" (CERT_E_CHAINING)");
    else if (r == (LONG)0x800B0003)     printf(" (TRUST_E_SUBJECT_FORM_UNKNOWN)");
    printf("\n");
}

// Dynamically-loaded CryptCAT types (mscat.h not in MinGW headers)
typedef void* HCATADMIN;
typedef void* HCATINFO;
struct CRYPTCATMEMBER_STUB {
    DWORD cbStruct;
    LPWSTR pwszReferenceTag;
    LPWSTR pwszFileName;
    GUID   gSubjectType;
    DWORD  fdwMemberFlags;
    void*  pIndirectData;
    DWORD  dwCertVersion;
    DWORD  dwReserved;
    HANDLE hReserved;
    CRYPT_ATTR_BLOB sEncodedIndirectData;
    CRYPT_ATTR_BLOB sEncodedMemberInfo;
};
typedef HCATINFO (WINAPI* PFN_CryptCATOpen)(LPWSTR, DWORD, HCRYPTPROV, DWORD, DWORD);
typedef CRYPTCATMEMBER_STUB* (WINAPI* PFN_CryptCATEnumerateMember)(HCATINFO, CRYPTCATMEMBER_STUB*);
typedef BOOL (WINAPI* PFN_CryptCATClose)(HCATINFO);
typedef BOOL (WINAPI* PFN_CryptCATAdminAcquireContext)(HCATADMIN*, const GUID*, DWORD);
typedef BOOL (WINAPI* PFN_CryptCATAdminCalcHashFromFileHandle)(HANDLE, DWORD*, BYTE*, DWORD);
typedef BOOL (WINAPI* PFN_CryptCATAdminReleaseContext)(HCATADMIN, DWORD);

static void run_signing_diagnostic() {
    printf("\n=== Catalog signing diagnostic ===\n");
    fflush(stdout);

    // Find catalog file
    const wchar_t* cat_path = nullptr;
    for (int i = 0; CAT_SEARCH_PATHS[i]; i++) {
        if (GetFileAttributesW(CAT_SEARCH_PATHS[i]) != INVALID_FILE_ATTRIBUTES) {
            cat_path = CAT_SEARCH_PATHS[i];
            printf("Catalog: %ls\n", cat_path);
            break;
        }
    }
    if (!cat_path) {
        printf("No catalog file found at any search path\n");
        fflush(stdout);
        return;
    }

    // Test 1a: WinVerifyTrust on catalog file using generic file action
    {
        WINTRUST_FILE_INFO fi = {};
        fi.cbStruct      = sizeof(fi);
        fi.pcwszFilePath = cat_path;

        WINTRUST_DATA wd = {};
        wd.cbStruct            = sizeof(wd);
        wd.dwUIChoice          = WTD_UI_NONE;
        wd.fdwRevocationChecks = WTD_REVOKE_NONE;
        wd.dwUnionChoice       = WTD_CHOICE_FILE;
        wd.pFile               = &fi;
        wd.dwStateAction       = WTD_STATEACTION_VERIFY;

        GUID guid = WINTRUST_ACTION_GENERIC_VERIFY_V2;
        LONG r = WinVerifyTrust(NULL, &guid, &wd);
        printf("WinVerifyTrust(file action)   = "); print_verify_result(r);

        wd.dwStateAction = WTD_STATEACTION_CLOSE;
        WinVerifyTrust(NULL, &guid, &wd);
    }

    // Test 1b: WinVerifyTrust using DRIVER_ACTION_VERIFY + catalog context (mirrors libcdsprpc.dll)
    {
        // WINTRUST_CATALOG_INFO structure (manually defined - not in MinGW headers)
        struct WINTRUST_CATALOG_INFO_STUB {
            DWORD   cbStruct;
            DWORD   dwCatalogVersion;
            LPCWSTR pcwszCatalogFilePath;
            LPCWSTR pcwszMemberTag;
            LPCWSTR pcwszMemberFilePath;
            HANDLE  hMemberFile;
            BYTE*   pbCalculatedFileHash;
            DWORD   cbCalculatedFileHash;
            PCCTL_CONTEXT pcCatalogContext;
            HCATADMIN hCatAdmin;
        };

        WINTRUST_CATALOG_INFO_STUB ci = {};
        ci.cbStruct             = sizeof(ci);
        ci.pcwszCatalogFilePath = cat_path;
        ci.pcwszMemberFilePath  = SKEL_PATH;
        ci.pcwszMemberTag       = nullptr; // let WinVerifyTrust compute hash from file

        WINTRUST_DATA wd = {};
        wd.cbStruct            = sizeof(wd);
        wd.dwUIChoice          = WTD_UI_NONE;
        wd.fdwRevocationChecks = WTD_REVOKE_NONE;
        wd.dwUnionChoice       = 2; // WTD_CHOICE_CATALOG
        wd.pCatalog            = (WINTRUST_CATALOG_INFO*)&ci;
        wd.dwStateAction       = WTD_STATEACTION_VERIFY;

        // DRIVER_ACTION_VERIFY = {F750E6C3-38EE-11D1-85E5-00C04FC295EE}
        GUID drvGuid;
        drvGuid.Data1=0xF750E6C3; drvGuid.Data2=0x38EE; drvGuid.Data3=0x11D1;
        drvGuid.Data4[0]=0x85; drvGuid.Data4[1]=0xE5; drvGuid.Data4[2]=0x00; drvGuid.Data4[3]=0xC0;
        drvGuid.Data4[4]=0x4F; drvGuid.Data4[5]=0xC2; drvGuid.Data4[6]=0x95; drvGuid.Data4[7]=0xEE;

        LONG r = WinVerifyTrust(NULL, &drvGuid, &wd);
        printf("WinVerifyTrust(driver/catalog) = "); print_verify_result(r);

        wd.dwStateAction = WTD_STATEACTION_CLOSE;
        WinVerifyTrust(NULL, &drvGuid, &wd);
    }

    // Test 1c: GENERIC_VERIFY_V2 + catalog context (third combination)
    {
        struct WINTRUST_CATALOG_INFO_STUB2 {
            DWORD   cbStruct;
            DWORD   dwCatalogVersion;
            LPCWSTR pcwszCatalogFilePath;
            LPCWSTR pcwszMemberTag;
            LPCWSTR pcwszMemberFilePath;
            HANDLE  hMemberFile;
            BYTE*   pbCalculatedFileHash;
            DWORD   cbCalculatedFileHash;
            PCCTL_CONTEXT pcCatalogContext;
            HCATADMIN hCatAdmin;
        };
        WINTRUST_CATALOG_INFO_STUB2 ci2 = {};
        ci2.cbStruct             = sizeof(ci2);
        ci2.pcwszCatalogFilePath = cat_path;
        ci2.pcwszMemberFilePath  = SKEL_PATH;
        ci2.pcwszMemberTag       = nullptr;

        WINTRUST_DATA wd2 = {};
        wd2.cbStruct            = sizeof(wd2);
        wd2.dwUIChoice          = WTD_UI_NONE;
        wd2.fdwRevocationChecks = WTD_REVOKE_NONE;
        wd2.dwUnionChoice       = 2;
        wd2.pCatalog            = (WINTRUST_CATALOG_INFO*)&ci2;
        wd2.dwStateAction       = WTD_STATEACTION_VERIFY;

        GUID genGuid = WINTRUST_ACTION_GENERIC_VERIFY_V2;
        LONG r2 = WinVerifyTrust(NULL, &genGuid, &wd2);
        printf("WinVerifyTrust(generic/catalog) = "); print_verify_result(r2);

        wd2.dwStateAction = WTD_STATEACTION_CLOSE;
        WinVerifyTrust(NULL, &genGuid, &wd2);
    }

    // Test 2: Load CryptCAT functions dynamically and enumerate catalog members
    {
        HMODULE hWintrust = LoadLibraryA("wintrust.dll");
        if (!hWintrust) {
            printf("LoadLibrary(wintrust.dll) failed\n");
        } else {
            auto fnOpen    = (PFN_CryptCATOpen)GetProcAddress(hWintrust, "CryptCATOpen");
            auto fnEnum    = (PFN_CryptCATEnumerateMember)GetProcAddress(hWintrust, "CryptCATEnumerateMember");
            auto fnClose   = (PFN_CryptCATClose)GetProcAddress(hWintrust, "CryptCATClose");
            auto fnAcquire = (PFN_CryptCATAdminAcquireContext)GetProcAddress(hWintrust, "CryptCATAdminAcquireContext");
            auto fnHash    = (PFN_CryptCATAdminCalcHashFromFileHandle)GetProcAddress(hWintrust, "CryptCATAdminCalcHashFromFileHandle");
            auto fnRelease = (PFN_CryptCATAdminReleaseContext)GetProcAddress(hWintrust, "CryptCATAdminReleaseContext");

            if (fnOpen && fnEnum && fnClose) {
                // 0x4 = CRYPTCAT_OPEN_EXISTING, 0x100 = CRYPTCAT_VERSION_1
                HCATINFO hCat = fnOpen((LPWSTR)cat_path, 0x4, 0, 0x100, 0);
                if (!hCat || hCat == INVALID_HANDLE_VALUE) {
                    printf("CryptCATOpen failed: 0x%lx\n", (unsigned long)GetLastError());
                } else {
                    printf("CryptCATOpen: OK\n");
                    CRYPTCATMEMBER_STUB* m = nullptr;
                    int n = 0;
                    while ((m = fnEnum(hCat, m)) != nullptr) {
                        n++;
                        printf("  member[%d] tag: %ls\n", n,
                               m->pwszReferenceTag ? m->pwszReferenceTag : L"(null)");
                    }
                    printf("Total catalog members: %d\n", n);
                    fnClose(hCat);
                }
            } else {
                printf("CryptCAT functions not found in wintrust.dll\n");
            }

            // Test 3: Skel file hashes (System32\unknown AND ADSP_LIBRARY_PATH)
            printf("Skel: %ls -> ", SKEL_PATH);
            if (GetFileAttributesW(SKEL_PATH) == INVALID_FILE_ATTRIBUTES) {
                printf("NOT FOUND\n");
            } else {
                printf("EXISTS\n");
                if (fnAcquire && fnHash && fnRelease) {
                    HANDLE hFile = CreateFileW(SKEL_PATH, GENERIC_READ, FILE_SHARE_READ,
                                               nullptr, OPEN_EXISTING, 0, nullptr);
                    if (hFile != INVALID_HANDLE_VALUE) {
                        HCATADMIN hAdmin = nullptr;
                        GUID dg;
                        // DRIVER_ACTION_VERIFY = {F750E6C3-38EE-11D1-85E5-00C04FC295EE}
                        dg.Data1=0xF750E6C3; dg.Data2=0x38EE; dg.Data3=0x11D1;
                        dg.Data4[0]=0x85; dg.Data4[1]=0xE5; dg.Data4[2]=0x00; dg.Data4[3]=0xC0;
                        dg.Data4[4]=0x4F; dg.Data4[5]=0xC2; dg.Data4[6]=0x95; dg.Data4[7]=0xEE;
                        if (fnAcquire(&hAdmin, &dg, 0)) {
                            DWORD hashLen = 0;
                            fnHash(hFile, &hashLen, nullptr, 0);
                            if (hashLen > 0) {
                                BYTE* hash = (BYTE*)malloc(hashLen);
                                if (fnHash(hFile, &hashLen, hash, 0)) {
                                    printf("Skel hash (%lu bytes): ", (unsigned long)hashLen);
                                    for (DWORD i = 0; i < hashLen; i++) printf("%02x", hash[i]);
                                    printf("\n");
                                }
                                free(hash);
                            }
                            fnRelease(hAdmin, 0);
                        }
                        CloseHandle(hFile);
                    }
                }
            }

            // Hash the skel at ADSP_LIBRARY_PATH too
            char adsp_buf[512] = {};
            GetEnvironmentVariableA("ADSP_LIBRARY_PATH", adsp_buf, sizeof(adsp_buf));
            const char* adsp = adsp_buf[0] ? adsp_buf : nullptr;
            if (adsp && fnAcquire && fnHash && fnRelease) {
                char adsp_skel[512];
                snprintf(adsp_skel, sizeof(adsp_skel), "%s\\libggml-htp-v73.so", adsp);
                int wlen = MultiByteToWideChar(CP_ACP, 0, adsp_skel, -1, nullptr, 0);
                wchar_t* wadsp = (wchar_t*)malloc(wlen * sizeof(wchar_t));
                MultiByteToWideChar(CP_ACP, 0, adsp_skel, -1, wadsp, wlen);
                printf("ADSP skel: %ls -> ", wadsp);
                if (GetFileAttributesW(wadsp) == INVALID_FILE_ATTRIBUTES) {
                    printf("NOT FOUND\n");
                } else {
                    HANDLE hf = CreateFileW(wadsp, GENERIC_READ, FILE_SHARE_READ,
                                            nullptr, OPEN_EXISTING, 0, nullptr);
                    if (hf != INVALID_HANDLE_VALUE) {
                        HCATADMIN hAdm = nullptr;
                        GUID dg2;
                        dg2.Data1=0xF750E6C3; dg2.Data2=0x38EE; dg2.Data3=0x11D1;
                        dg2.Data4[0]=0x85; dg2.Data4[1]=0xE5; dg2.Data4[2]=0x00; dg2.Data4[3]=0xC0;
                        dg2.Data4[4]=0x4F; dg2.Data4[5]=0xC2; dg2.Data4[6]=0x95; dg2.Data4[7]=0xEE;
                        if (fnAcquire(&hAdm, &dg2, 0)) {
                            DWORD hl2 = 0;
                            fnHash(hf, &hl2, nullptr, 0);
                            if (hl2 > 0) {
                                BYTE* h2 = (BYTE*)malloc(hl2);
                                if (fnHash(hf, &hl2, h2, 0)) {
                                    printf("hash: ");
                                    for (DWORD i = 0; i < hl2; i++) printf("%02x", h2[i]);
                                    printf("\n");
                                }
                                free(h2);
                            }
                            fnRelease(hAdm, 0);
                        }
                        CloseHandle(hf);
                    }
                }
                free(wadsp);
            }
            FreeLibrary(hWintrust);
        }
    }

    // Test 4: ELF header check on skel file (should be Hexagon DSP, machine=0xA4)
    {
        HANDLE hf = CreateFileW(SKEL_PATH, GENERIC_READ, FILE_SHARE_READ,
                                nullptr, OPEN_EXISTING, 0, nullptr);
        if (hf != INVALID_HANDLE_VALUE) {
            unsigned char hdr[20] = {};
            DWORD nread = 0;
            if (ReadFile(hf, hdr, sizeof(hdr), &nread, nullptr) && nread >= 20) {
                bool isElf = (hdr[0]==0x7f && hdr[1]=='E' && hdr[2]=='L' && hdr[3]=='F');
                int elfClass = hdr[4]; // 1=32-bit, 2=64-bit
                int machine  = hdr[18] | (hdr[19] << 8);
                printf("Skel ELF: magic=%s class=%d machine=0x%02x (%s)\n",
                       isElf ? "OK" : "BAD",
                       elfClass,
                       machine,
                       machine == 0xA4 ? "Hexagon DSP - CORRECT" :
                       machine == 0xB7 ? "AArch64 - WRONG!" :
                       machine == 0x28 ? "ARM32 - WRONG!" : "unknown");
            }
            CloseHandle(hf);
        }
    }

    // Test 5: SHA-256 hash of skel (should match one of the SHA-256 catalog entries)
    {
        HMODULE hBcrypt = LoadLibraryA("bcrypt.dll");
        if (hBcrypt) {
            typedef LONG (WINAPI* PFN_BCryptOpenAlgorithmProvider)(void**, LPCWSTR, LPCWSTR, ULONG);
            typedef LONG (WINAPI* PFN_BCryptCreateHash)(void*, void**, BYTE*, ULONG, BYTE*, ULONG, ULONG);
            typedef LONG (WINAPI* PFN_BCryptHashData)(void*, BYTE*, ULONG, ULONG);
            typedef LONG (WINAPI* PFN_BCryptFinishHash)(void*, BYTE*, ULONG, ULONG);
            typedef LONG (WINAPI* PFN_BCryptDestroyHash)(void*);
            typedef LONG (WINAPI* PFN_BCryptCloseAlgorithmProvider)(void*, ULONG);
            typedef LONG (WINAPI* PFN_BCryptGetProperty)(void*, LPCWSTR, BYTE*, ULONG, ULONG*, ULONG);

            auto fnOpen    = (PFN_BCryptOpenAlgorithmProvider)GetProcAddress(hBcrypt, "BCryptOpenAlgorithmProvider");
            auto fnCreate  = (PFN_BCryptCreateHash)GetProcAddress(hBcrypt, "BCryptCreateHash");
            auto fnData    = (PFN_BCryptHashData)GetProcAddress(hBcrypt, "BCryptHashData");
            auto fnFinish  = (PFN_BCryptFinishHash)GetProcAddress(hBcrypt, "BCryptFinishHash");
            auto fnDestroy = (PFN_BCryptDestroyHash)GetProcAddress(hBcrypt, "BCryptDestroyHash");
            auto fnClose   = (PFN_BCryptCloseAlgorithmProvider)GetProcAddress(hBcrypt, "BCryptCloseAlgorithmProvider");
            auto fnGetProp = (PFN_BCryptGetProperty)GetProcAddress(hBcrypt, "BCryptGetProperty");

            if (fnOpen && fnCreate && fnData && fnFinish && fnDestroy && fnClose && fnGetProp) {
                void* hAlg = nullptr;
                if (fnOpen(&hAlg, L"SHA256", nullptr, 0) == 0) {
                    ULONG hashLen = 0, cbResult = 0;
                    fnGetProp(hAlg, L"HashDigestLength", (BYTE*)&hashLen, sizeof(hashLen), &cbResult, 0);
                    void* hHash = nullptr;
                    if (fnCreate(hAlg, &hHash, nullptr, 0, nullptr, 0, 0) == 0) {
                        HANDLE hf = CreateFileW(SKEL_PATH, GENERIC_READ, FILE_SHARE_READ,
                                                nullptr, OPEN_EXISTING, 0, nullptr);
                        if (hf != INVALID_HANDLE_VALUE) {
                            BYTE buf[4096];
                            DWORD n;
                            while (ReadFile(hf, buf, sizeof(buf), &n, nullptr) && n > 0)
                                fnData(hHash, buf, n, 0);
                            CloseHandle(hf);
                        }
                        BYTE* digest = (BYTE*)malloc(hashLen);
                        if (fnFinish(hHash, digest, hashLen, 0) == 0) {
                            printf("Skel SHA-256: ");
                            for (ULONG i = 0; i < hashLen; i++) printf("%02x", digest[i]);
                            printf("\n");
                        }
                        free(digest);
                        fnDestroy(hHash);
                    }
                    fnClose(hAlg, 0);
                }
            }
            FreeLibrary(hBcrypt);
        }
    }

    printf("=== End signing diagnostic ===\n\n");
    fflush(stdout);
}

int main(int argc, char** argv) {
    printf("=== FastRPC driver test ===\n"); fflush(stdout);

    // If ADSP_LIBRARY_PATH not set, set it from argv[1] or a default
    const char* adsp_path = getenv("ADSP_LIBRARY_PATH");
    if (!adsp_path) {
        const char* default_skel_dir =
            (argc > 1) ? argv[1]
                       : "C:\\Users\\justi\\GitRepos\\qualcomm-npu\\third_party\\llama.cpp\\pkg-snapdragon\\lib";
        SetEnvironmentVariableA("ADSP_LIBRARY_PATH", default_skel_dir);
        adsp_path = default_skel_dir;
    }
    printf("ADSP_LIBRARY_PATH=%s\n", adsp_path); fflush(stdout);
    run_signing_diagnostic();

    // 1. Init driver (loads libcdsprpc.dll)
    AEEResult err = fastrpc_drv_init();
    if (err != AEE_SUCCESS) {
        printf("FAIL: fastrpc_drv_init() = 0x%x\n", err);
        return 1;
    }
    printf("PASS: driver loaded\n"); fflush(stdout);

    // 2. Query arch version
    int arch = 0;
    int r = fastrpc_drv_get_arch(&arch);
    if (r != 0) {
        printf("FAIL: get_arch = %d\n", r);
        return 1;
    }
    printf("PASS: Hexagon arch = v%d\n", arch); fflush(stdout);

    void* buf = nullptr;
    int fd = -1;

    // 4b. Match llama.cpp dev_id=0 path: NO RESERVE_NEW_SESSION, use default session.
    // llama.cpp only calls RESERVE_NEW_SESSION for dev_id > 0 (additional devices).
    // For dev_id=0: session_id=0, domain_id=3, dspqueue on domain=3.
    uint32_t effective_domain_id = 3;
    uint32_t reserved_session_id = 0;

    // Get session URI via FASTRPC_GET_URI (matches llama.cpp exactly)
    char htp_uri[256];
    snprintf(htp_uri, sizeof(htp_uri),
             "file:///libggml-htp-v%d.so?htp_iface_skel_handle_invoke&_modver=1.0", arch);

    char session_uri[256];
    {
        struct remote_rpc_get_uri u = {};
        u.session_id      = reserved_session_id;
        u.domain_name     = const_cast<char*>(CDSP_DOMAIN_NAME);
        u.domain_name_len = (uint32_t)strlen(CDSP_DOMAIN_NAME);
        u.module_uri      = htp_uri;
        u.module_uri_len  = (uint32_t)strlen(htp_uri);
        u.uri             = session_uri;
        u.uri_len         = sizeof(session_uri);
        err = remote_session_control(FASTRPC_GET_URI, &u, sizeof(u));
        if (err != AEE_SUCCESS) {
            // fallback: raw URI + domain suffix
            snprintf(session_uri, sizeof(session_uri), "%s%s", htp_uri, CDSP_DOMAIN_NAME);
            printf("INFO: FASTRPC_GET_URI failed 0x%x, fallback URI: %s\n", err, session_uri);
        } else {
            printf("PASS: FASTRPC_GET_URI ok, uri=%s\n", session_uri);
        }
        fflush(stdout);
    }

    // Probe: try a Qualcomm-signed skel to confirm FastRPC catalog verification works at all
    {
        remote_handle64 qc_handle = 0;
        const char* qc_uri = "file:///libbenchmark_skel.so?benchmark_skel_handle_invoke&_modver=1.0&_dom=cdsp&_session=0";
        AEEResult qc_err = htp_iface_open(qc_uri, &qc_handle);
        if (qc_err == AEE_SUCCESS) {
            printf("INFO: Qualcomm benchmark skel opened OK (catalog works)\n");
            htp_iface_close(qc_handle);
        } else {
            printf("INFO: Qualcomm benchmark skel = 0x%x (expected if not in ADSP_LIBRARY_PATH)\n", qc_err);
        }
        fflush(stdout);
    }

    // Enable unsigned PD (domain=3)
    fprintf(stderr, "[test] calling DSPRPC_CONTROL_UNSIGNED_MODULE\n");
    {
        struct remote_rpc_control_unsigned_module u;
        u.domain = (int)effective_domain_id;
        u.enable = 1;
        err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &u, sizeof(u));
        printf("INFO: DSPRPC_CONTROL_UNSIGNED_MODULE(domain=%u) = 0x%x %s\n",
               effective_domain_id, err, err == AEE_SUCCESS ? "(ok)" : "(failed)");
        fflush(stdout);
    }

    // 6b. Open FastRPC session
    fprintf(stderr, "[test] calling htp_iface_open\n");
    remote_handle64 handle = 0;
    err = htp_iface_open(session_uri, &handle);
    if (err != AEE_SUCCESS) {
        printf("FAIL: htp_iface_open = 0x%x\n", err);
        return 1;
    }
    printf("PASS: session opened, handle=0x%llx\n", (unsigned long long)handle); fflush(stdout);

    // Enable FastRPC QoS mode (matches llama.cpp sequence)
    {
        struct remote_rpc_control_latency l;
        l.enable = 1;
        int qos_err = remote_handle64_control(handle, DSPRPC_CONTROL_LATENCY, &l, sizeof(l));
        printf("INFO: remote_handle64_control(LATENCY) = 0x%x %s\n", qos_err,
               qos_err == 0 ? "(ok)" : "(failed, non-fatal)");
        fflush(stdout);
    }

    // 7. Create dspqueue
    // Queue sizes matching llama.cpp: (sizeof(req) * 16 * 2) + 1024, same for rsp
    // htp_opbatch_req = 16 bytes, htp_opbatch_rsp = 4 bytes
    const uint32_t REQ_Q_SZ = (16 * 1024 * 2) + 1024;  // 33792 (matches llama-bench opt_opqueue=1024)
    const uint32_t RSP_Q_SZ = (4  * 1024 * 2) + 1024;  // 9216
    // dspqueue_create on effective_domain_id (3 for default session, matching llama.cpp dev_id=0).
    dspqueue_t queue = nullptr;
    fprintf(stderr, "[test] calling dspqueue_create(domain=%u)\n", effective_domain_id); fflush(stderr);
    err = dspqueue_create((int)effective_domain_id, 0, REQ_Q_SZ, RSP_Q_SZ,
                          nullptr, nullptr, nullptr, &queue);
    printf("INFO: dspqueue_create(domain=%u) = 0x%x\n", effective_domain_id, err); fflush(stdout);
    if (err != AEE_SUCCESS) {
        printf("FAIL: dspqueue_create = 0x%x\n", err);
        htp_iface_close(handle);
        rpcmem_free(buf);
        return 1;
    }
    printf("PASS: dspqueue created\n");

    // 7. Export queue ID
    uint64_t queue_id = 0;
    fprintf(stderr, "[test] calling dspqueue_export\n"); fflush(stderr);
    err = dspqueue_export(queue, &queue_id);
    if (err != AEE_SUCCESS) {
        printf("FAIL: dspqueue_export = 0x%x\n", err);
    } else {
        printf("PASS: queue_id = %llu\n", (unsigned long long)queue_id);
    }
    fflush(stdout);

    // 8. Start DSP session
    fprintf(stderr, "[test] calling htp_iface_start\n"); fflush(stderr);
    err = htp_iface_start(handle, (uint32_t)reserved_session_id, queue_id, 0, 1);
    printf("INFO: htp_iface_start = 0x%x %s\n", err, err == AEE_SUCCESS ? "(PASS)" : "(FAIL)");
    fflush(stdout);

    // 9. Map rpcmem into DSP (post-start)
    fprintf(stderr, "[test] calling htp_iface_mmap (post-start)\n"); fflush(stderr);
    err = htp_iface_mmap(handle, (uint32_t)fd, 4096, 0);
    printf("INFO: htp_iface_mmap (post-start) = 0x%x %s\n", err, err == AEE_SUCCESS ? "(PASS)" : "(FAIL)");
    fflush(stdout);

    // 10. Send a trivial op batch (just ADD of two zero tensors) as a smoke test
    // This exercises the full dspqueue round-trip.
    printf("Testing dspqueue round-trip...\n");

    // Allocate scratch for batch payload
    const size_t SHM_SIZE = 64 * 1024;
    void* shm = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (int)SHM_SIZE);
    int shm_fd = rpcmem_alloc ? rpcmem_to_fd(shm) : -1;
    if (!shm || shm_fd < 0) {
        printf("FAIL: shm alloc failed\n");
        goto cleanup;
    }
    err = htp_iface_mmap(handle, (uint32_t)shm_fd, (uint32_t)SHM_SIZE, 0);
    if (err != AEE_SUCCESS) {
        printf("FAIL: shm mmap = 0x%x\n", err);
        goto cleanup;
    }

    {
        // 4 elements of FP16
        const uint32_t N = 4;
        memset(buf, 0, 4096);

        htp_buf_desc  bufs[2]    = {};
        htp_tensor    tensors[3] = {};
        htp_op_desc   ops[1]     = {};

        bufs[0].base = (uint64_t)(uintptr_t)buf;
        bufs[0].size = 4096;
        bufs[0].fd   = (uint32_t)fd;

        bufs[1].base = (uint64_t)(uintptr_t)shm;
        bufs[1].size = SHM_SIZE;
        bufs[1].fd   = (uint32_t)shm_fd;

        // src0: [N] FP16 at buf offset 0
        tensors[0].bi   = 0; tensors[0].data = 0; tensors[0].size = N*2;
        tensors[0].type = HTP_TYPE_F16; tensors[0].ne[0] = N; tensors[0].ne[1] = 1;
        tensors[0].nb[0] = 2; tensors[0].nb[1] = N*2;

        // src1: [N] FP16 at buf offset N*2
        tensors[1].bi   = 0; tensors[1].data = N*2; tensors[1].size = N*2;
        tensors[1].type = HTP_TYPE_F16; tensors[1].ne[0] = N; tensors[1].ne[1] = 1;
        tensors[1].nb[0] = 2; tensors[1].nb[1] = N*2;

        // dst: [N] FP16 at buf offset N*4
        tensors[2].bi   = 0; tensors[2].data = N*4; tensors[2].size = N*2;
        tensors[2].type = HTP_TYPE_F16; tensors[2].flags = HTP_TENSOR_COMPUTE;
        tensors[2].ne[0] = N; tensors[2].ne[1] = 1;
        tensors[2].nb[0] = 2; tensors[2].nb[1] = N*2;

        // ADD: dst = src0 + src1
        ops[0].opcode = HTP_OP_ADD;
        ops[0].src[0] = 0; ops[0].src[1] = 1;
        ops[0].dst    = 2;

        // Pack into shm
        uint8_t* p = (uint8_t*)shm;
        memcpy(p,                                     bufs,    sizeof(bufs));    p += sizeof(bufs);
        memcpy(p,                                     tensors, sizeof(tensors)); p += sizeof(tensors);
        memcpy(p,                                     ops,     sizeof(ops));
        size_t payload = sizeof(bufs) + sizeof(tensors) + sizeof(ops);

        htp_opbatch_req req = { 2, 3, 1, 0 };
        struct dspqueue_buffer dbuf = {};
        dbuf.fd     = shm_fd;
        dbuf.flags  = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
        dbuf.ptr    = shm;
        dbuf.offset = 0;
        dbuf.size   = (uint32_t)payload;

        err = dspqueue_write(queue, 0, 1, &dbuf,
                             sizeof(req), (const uint8_t*)&req,
                             DSPQUEUE_TIMEOUT);
        if (err != AEE_SUCCESS) {
            printf("FAIL: dspqueue_write = 0x%x\n", err);
            goto cleanup;
        }

        htp_opbatch_rsp rsp = {};
        struct dspqueue_buffer rbuf = {};
        uint32_t flg = 0, nb = 0, ml = 0;
        err = dspqueue_read(queue, &flg, 1, &nb, &rbuf,
                            sizeof(rsp), &ml, (uint8_t*)&rsp,
                            DSPQUEUE_TIMEOUT);
        if (err != AEE_SUCCESS) {
            printf("FAIL: dspqueue_read = 0x%x\n", err);
            goto cleanup;
        }
        if (rsp.status != HTP_STATUS_OK) {
            printf("FAIL: DSP op status = %u\n", rsp.status);
        } else {
            printf("PASS: dspqueue round-trip OK (status=HTP_STATUS_OK)\n");
        }
    }

cleanup:
    if (shm) { htp_iface_munmap(handle, shm_fd); rpcmem_free(shm); }
    if (queue) { htp_iface_stop(handle); dspqueue_close(queue); }
    if (handle) htp_iface_close(handle);
    rpcmem_free(buf);
    printf("=== done ===\n");
    return 0;
}
