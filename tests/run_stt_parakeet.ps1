$env:CACTUS_TEST_TRANSCRIBE_MODEL = 'C:\Users\justi\GitRepos\qualcomm-npu\third_party\cactus\weights\parakeet-ctc-1.1b'
$env:CACTUS_TEST_ASSETS = 'C:\Users\justi\GitRepos\qualcomm-npu\third_party\cactus\tests\assets'
$env:CACTUS_PARAKEET_T = '400'
$env:PATH = 'C:\msys64\clangarm64\bin;' + $env:PATH
& 'C:\Users\justi\GitRepos\qualcomm-npu\third_party\cactus\tests\build\test_stt.exe' `
    'C:\Users\justi\GitRepos\qualcomm-npu\third_party\cactus\weights\parakeet-ctc-1.1b' `
    'C:\Users\justi\GitRepos\qualcomm-npu\third_party\cactus\tests\assets\test.wav'
