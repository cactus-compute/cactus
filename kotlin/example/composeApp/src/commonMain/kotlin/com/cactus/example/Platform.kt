package com.cactus.example

expect suspend fun saveImageToTempFile(imageBytes: ByteArray): String?
expect suspend fun saveAudioToTempFile(audioBytes: ByteArray): String?
