# cactus_example

Flutter 예제 프로젝트입니다.

## 시작하기

먼저 `git clone https://github.com/cactus-compute/cactus.git`으로 저장소를 복제하고, 해당 디렉토리로 이동한 후 `chmod +x scripts/*.sh`로 모든 스크립트를 실행 가능하게 만드세요.

- `scripts/build-flutter-android.sh`를 실행하여 Android JNILibs를 빌드합니다.
- `scripts/build-flutter.sh`를 실행하여 Flutter 플러그인을 빌드합니다. (예제 사용 전 반드시 실행)
- `cd flutter/example`로 예제 앱으로 이동합니다.
- Xcode 또는 Android Studio를 통해 시뮬레이터를 엽니다. 처음이시라면 [가이드](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda)를 참조하세요.
- 항상 `flutter clean && flutter pub get && flutter run` 조합으로 앱을 시작합니다.
- 앱을 실행하고 원하는 대로 예제 앱이나 플러그인을 수정해보세요.

## 리소스

Flutter 프로젝트가 처음이시라면 다음 리소스를 참고하세요:

- [Lab: 첫 번째 Flutter 앱 작성하기](https://docs.flutter.dev/get-started/codelab)
- [Cookbook: 유용한 Flutter 샘플](https://docs.flutter.dev/cookbook)

Flutter 개발을 시작하는 데 도움이 필요하시면
[온라인 문서](https://docs.flutter.dev/)를 참조하세요. 튜토리얼,
샘플, 모바일 개발 가이드, 전체 API 레퍼런스를 제공합니다.