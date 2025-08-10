@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)
package com.cactus.example

import kotlinx.cinterop.*
import platform.Foundation.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

actual suspend fun saveImageToTempFile(imageBytes: ByteArray): String? {
    return withContext(Dispatchers.Default) {
        try {
            val documentsDir = NSSearchPathForDirectoriesInDomains(
                NSDocumentDirectory, NSUserDomainMask, true
            ).firstOrNull() as? String ?: return@withContext null
            
            val targetPath = "$documentsDir/temp_image.jpg"
            
            val data = imageBytes.toNSData()
            data.writeToFile(targetPath, true)
            
            targetPath
        } catch (e: Exception) {
            null
        }
    }
}

actual suspend fun saveAudioToTempFile(audioBytes: ByteArray): String? {
    return withContext(Dispatchers.Default) {
        try {
            val documentsDir = NSSearchPathForDirectoriesInDomains(
                NSDocumentDirectory, NSUserDomainMask, true
            ).firstOrNull() as? String ?: return@withContext null

            val targetPath = "$documentsDir/temp_audio.wav"

            val data = audioBytes.toNSData()
            data.writeToFile(targetPath, true)

            targetPath
        } catch (e: Exception) {
            null
        }
    }
}

@OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)
private fun ByteArray.toNSData(): NSData = memScoped {
    NSData.create(bytes = allocArrayOf(this@toNSData), length = this@toNSData.size.toULong())
}
