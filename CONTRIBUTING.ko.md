# Cactus 기여 가이드

Cactus 프로젝트에 기여해 주셔서 감사합니다! 이 가이드는 프로젝트에 기여하는 방법을 설명합니다.

## 📋 목차

- [시작하기](#시작하기)
- [기여 방법](#기여-방법)
- [개발 환경 설정](#개발-환경-설정)
- [코드 스타일](#코드-스타일)
- [Pull Request 프로세스](#pull-request-프로세스)
- [이슈 보고](#이슈-보고)
- [커뮤니티](#커뮤니티)

## 시작하기

1. **저장소 포크하기**: GitHub에서 저장소를 포크합니다.
2. **저장소 복제하기**: 
   ```bash
   git clone https://github.com/your-username/cactus.git
   cd cactus
   ```
3. **스크립트 실행 권한 부여**: 
   ```bash
   chmod +x scripts/*.sh
   ```

## 기여 방법

### 버그 수정
- 버그를 발견하면 먼저 [이슈 트래커](https://github.com/cactus-compute/cactus/issues)에서 이미 보고되었는지 확인하세요.
- 새로운 브랜치를 생성합니다: `git checkout -b fix-bug-description`
- 수정 사항을 구현합니다.
- 테스트를 실행하여 수정 사항이 올바르게 작동하는지 확인합니다.
- Pull Request를 제출합니다.

### 새로운 기능
- **중요**: 새로운 기능을 구현하기 전에 먼저 [이슈를 생성](https://github.com/cactus-compute/cactus/issues/new)하여 논의해주세요.
- 이는 다른 기여자와의 작업 중복을 방지하고, 기능이 프로젝트 방향과 일치하는지 확인하기 위함입니다.
- 승인을 받은 후 구현을 시작하세요.

### 문서 개선
- README, 가이드, API 문서 등의 개선을 환영합니다.
- 번역 기여도 환영합니다!

## 개발 환경 설정

### Flutter 개발
```bash
# Android JNILibs 빌드
./scripts/build-flutter-android.sh

# Flutter 플러그인 빌드
./scripts/build-flutter.sh

# 예제 앱으로 이동
cd flutter/example

# 앱 실행
flutter clean && flutter pub get && flutter run
```

### React Native 개발
```bash
# Android JNILibs 빌드
./scripts/build-react-android.sh

# React Native 패키지 빌드
./scripts/build-react.sh

# 예제 앱으로 이동
cd react/example

# 의존성 설치 및 실행
yarn && yarn ios  # 또는 yarn android
```

### Kotlin Multiplatform 개발
```bash
# Android JNILibs 빌드 (Flutter와 공유)
./scripts/build-flutter-android.sh

# Kotlin 라이브러리 빌드
./scripts/build-kotlin.sh

# 예제 앱으로 이동
cd kotlin/example

# 데스크톱 앱 실행
./gradlew :composeApp:run
```

### C++ 개발
```bash
# 예제 디렉토리로 이동
cd cpp/example

# 빌드
./build.sh

# 실행
./cactus_llm  # 또는 cactus_vlm, cactus_embed, cactus_tts
```

## 코드 스타일

### 일반 규칙
- 기존 코드 스타일을 따라주세요
- 의미 있는 변수명과 함수명을 사용하세요
- 복잡한 로직에는 주석을 추가하세요

### 언어별 규칙

#### Dart/Flutter
- `flutter analyze`를 실행하여 린트 오류가 없는지 확인하세요
- `flutter format`을 사용하여 코드를 포맷팅하세요

#### JavaScript/TypeScript
- ESLint 규칙을 따라주세요
- Prettier를 사용하여 코드를 포맷팅하세요

#### Kotlin
- Kotlin 코딩 규칙을 따라주세요
- IDE의 자동 포맷팅 기능을 활용하세요

#### C++
- 기존 C++ 코드 스타일을 유지하세요
- 헤더 가드를 사용하세요

## Pull Request 프로세스

1. **브랜치 생성**: 설명적인 이름으로 브랜치를 생성합니다
   - 버그 수정: `fix-crash-on-model-load`
   - 기능 추가: `feature-add-korean-support`
   - 문서 개선: `docs-update-readme`

2. **커밋 메시지**: 명확하고 간결한 커밋 메시지를 작성합니다
   ```
   타입: 간단한 설명
   
   자세한 설명 (필요한 경우)
   
   Fixes #이슈번호 (해당하는 경우)
   ```

3. **테스트**: 모든 테스트가 통과하는지 확인합니다

4. **PR 생성**: 
   - 변경 사항을 명확히 설명합니다
   - 관련 이슈를 참조합니다
   - 스크린샷이나 GIF를 추가합니다 (UI 변경의 경우)

5. **코드 리뷰**: 리뷰어의 피드백에 응답하고 필요한 수정을 합니다

## 이슈 보고

이슈를 보고할 때 다음 정보를 포함해주세요:

### 버그 보고
- **설명**: 버그에 대한 명확한 설명
- **재현 방법**: 단계별 재현 방법
- **예상 동작**: 예상했던 동작
- **실제 동작**: 실제로 발생한 동작
- **환경**:
  - OS 및 버전
  - 플랫폼 (Flutter/React Native/Kotlin/C++)
  - Cactus 버전
  - 디바이스 정보 (모바일의 경우)

### 기능 요청
- **문제 설명**: 해결하려는 문제
- **제안 솔루션**: 제안하는 해결 방법
- **대안**: 고려한 다른 방법들
- **추가 정보**: 관련 자료나 예시

## 커뮤니티

- **Discord**: [Discord 서버 참여](https://discord.gg/bNurx3AXTJ)
- **GitHub Discussions**: 아이디어 논의 및 질문
- **이슈 트래커**: 버그 보고 및 기능 요청

## 행동 강령

모든 기여자는 다음을 준수해야 합니다:
- 서로 존중하고 예의 바르게 행동하기
- 건설적인 피드백 제공하기
- 다양성과 포용성 존중하기
- 프로젝트 목표에 집중하기

## 라이선스

기여한 코드는 프로젝트의 기존 라이선스를 따릅니다.

## 도움이 필요하신가요?

- [Discord 서버](https://discord.gg/bNurx3AXTJ)에서 질문하세요
- GitHub Issues에서 도움을 요청하세요
- 문서를 확인하세요

감사합니다! 여러분의 기여가 Cactus를 더 나은 프로젝트로 만듭니다. 🌵