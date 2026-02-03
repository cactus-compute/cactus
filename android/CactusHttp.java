package com.cactus;

import android.content.Context;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.util.Log;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.SocketTimeoutException;
import java.net.URL;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import javax.net.ssl.SSLException;

/**
 * HTTP helper class for Cactus cloud fallback.
 * Called from native C++ code via JNI for HTTP requests.
 */
public class CactusHttp {
    private static final String TAG = "CactusHttp";

    // Thread pool for HTTP requests
    private static final ExecutorService executor = Executors.newCachedThreadPool();

    // Maximum response body size (10 MB) to prevent OOM on huge responses
    private static final int MAX_RESPONSE_SIZE = 10 * 1024 * 1024;

    // Optional context for network connectivity check
    private static Context appContext = null;

    /**
     * Initialize with application context for network connectivity checks.
     * This is optional - if not called, network check will be skipped.
     *
     * @param context Application context
     */
    public static void init(Context context) {
        appContext = context.getApplicationContext();
    }

    /**
     * Check if network is available.
     * Returns true if context not set (assume available) or if network is connected.
     */
    public static boolean isNetworkAvailable() {
        if (appContext == null) {
            return true; // Assume available if context not provided
        }
        try {
            ConnectivityManager cm = (ConnectivityManager) appContext.getSystemService(Context.CONNECTIVITY_SERVICE);
            if (cm == null) return true;
            NetworkInfo activeNetwork = cm.getActiveNetworkInfo();
            return activeNetwork != null && activeNetwork.isConnected();
        } catch (Exception e) {
            Log.w(TAG, "Failed to check network connectivity: " + e.getMessage());
            return true; // Assume available on error
        }
    }

    /**
     * Native callback to deliver HTTP response back to C++.
     *
     * @param requestId Unique request identifier
     * @param statusCode HTTP status code (-1 for network errors)
     * @param body Response body (may be null on error)
     * @param error Error message (may be null on success)
     */
    private static native void nativeOnResponse(long requestId, int statusCode, String body, String error);

    /**
     * Perform HTTP POST request asynchronously.
     * Called from native C++ code.
     *
     * @param requestId Unique request identifier for response matching
     * @param url Target URL
     * @param body Request body (JSON)
     * @param contentType Content-Type header value
     * @param authorization Authorization header value (Bearer token or API key)
     * @param timeoutMs Request timeout in milliseconds
     */
    public static void post(final long requestId,
                            final String url,
                            final String body,
                            final String contentType,
                            final String authorization,
                            final int timeoutMs) {
        executor.execute(() -> {
            HttpURLConnection connection = null;
            int statusCode = -1;
            String responseBody = null;
            String error = null;

            try {
                // Optional network check
                if (!isNetworkAvailable()) {
                    error = "No network connection available";
                    nativeOnResponse(requestId, -1, null, error);
                    return;
                }

                URL urlObj = new URL(url);
                connection = (HttpURLConnection) urlObj.openConnection();
                connection.setRequestMethod("POST");
                connection.setConnectTimeout(timeoutMs);
                connection.setReadTimeout(timeoutMs);
                connection.setDoOutput(true);

                // Set headers
                connection.setRequestProperty("Content-Type", contentType != null ? contentType : "application/json");
                connection.setRequestProperty("Accept", "application/json");

                if (authorization != null && !authorization.isEmpty()) {
                    String sanitizedAuth = authorization.trim().replaceAll("[\\r\\n]", "");
                    connection.setRequestProperty("Authorization", sanitizedAuth);
                }

                // Write request body
                if (body != null && !body.isEmpty()) {
                    byte[] bodyBytes = body.getBytes(StandardCharsets.UTF_8);
                    connection.setRequestProperty("Content-Length", String.valueOf(bodyBytes.length));

                    try (OutputStream os = connection.getOutputStream()) {
                        os.write(bodyBytes);
                        os.flush();
                    }
                }

                // Get response
                statusCode = connection.getResponseCode();

                // Check Content-Type for captive portal detection
                // A captive portal returns 200 OK with text/html instead of application/json
                String contentTypeHeader = connection.getContentType();
                if (statusCode >= 200 && statusCode < 300 && contentTypeHeader != null) {
                    String lowerContentType = contentTypeHeader.toLowerCase();
                    if (!lowerContentType.contains("application/json") &&
                        !lowerContentType.contains("text/json")) {
                        // Likely a captive portal or error page
                        error = "Unexpected Content-Type: " + contentTypeHeader +
                                " (expected application/json). Possible captive portal or proxy interference.";
                        statusCode = -1;
                        nativeOnResponse(requestId, statusCode, null, error);
                        return;
                    }
                }

                // Read response body
                BufferedReader reader = null;
                try {
                    if (statusCode >= 200 && statusCode < 300) {
                        reader = new BufferedReader(new InputStreamReader(connection.getInputStream(), StandardCharsets.UTF_8));
                    } else {
                        // Handle null error stream gracefully
                        if (connection.getErrorStream() != null) {
                            reader = new BufferedReader(new InputStreamReader(connection.getErrorStream(), StandardCharsets.UTF_8));
                        }
                    }

                    if (reader != null) {
                        StringBuilder response = new StringBuilder();
                        String line;
                        int totalSize = 0;
                        while ((line = reader.readLine()) != null) {
                            totalSize += line.length();
                            if (totalSize > MAX_RESPONSE_SIZE) {
                                throw new IOException("Response body exceeds maximum size of " + MAX_RESPONSE_SIZE + " bytes");
                            }
                            response.append(line);
                        }
                        responseBody = response.toString();
                    }
                } finally {
                    if (reader != null) {
                        try { reader.close(); } catch (IOException ignored) {}
                    }
                }

                if (statusCode < 200 || statusCode >= 300) {
                    error = "HTTP " + statusCode + ": " + (responseBody != null ? responseBody : "Unknown error");
                }

            } catch (SocketTimeoutException e) {
                Log.e(TAG, "HTTP request timed out: " + e.getMessage());
                statusCode = -1;
                error = "Request timed out: " + e.getMessage();
            } catch (UnknownHostException e) {
                Log.e(TAG, "Unknown host: " + e.getMessage());
                statusCode = -1;
                error = "Unknown host (DNS resolution failed): " + e.getMessage();
            } catch (SSLException e) {
                Log.e(TAG, "SSL error: " + e.getMessage());
                statusCode = -1;
                error = "SSL/TLS error: " + e.getMessage();
            } catch (IOException e) {
                Log.e(TAG, "HTTP request failed: " + e.getMessage(), e);
                statusCode = -1;
                error = "Network error: " + e.getMessage();
            } catch (Exception e) {
                Log.e(TAG, "Unexpected error: " + e.getMessage(), e);
                statusCode = -1;
                error = "Error: " + e.getMessage();
            } finally {
                if (connection != null) {
                    try {
                        connection.disconnect();
                    } catch (Exception ignored) {}
                }
            }

            // Deliver response to native code
            try {
                nativeOnResponse(requestId, statusCode, responseBody, error);
            } catch (Exception e) {
                Log.e(TAG, "Failed to deliver response to native: " + e.getMessage(), e);
            }
        });
    }

    /**
     * Convenience method for simple GET request (not currently used by Cactus).
     */
    public static void get(final long requestId,
                           final String url,
                           final String authorization,
                           final int timeoutMs) {
        executor.execute(() -> {
            HttpURLConnection connection = null;
            int statusCode = -1;
            String responseBody = null;
            String error = null;

            try {
                if (!isNetworkAvailable()) {
                    error = "No network connection available";
                    nativeOnResponse(requestId, -1, null, error);
                    return;
                }

                URL urlObj = new URL(url);
                connection = (HttpURLConnection) urlObj.openConnection();
                connection.setRequestMethod("GET");
                connection.setConnectTimeout(timeoutMs);
                connection.setReadTimeout(timeoutMs);

                connection.setRequestProperty("Accept", "application/json");

                if (authorization != null && !authorization.isEmpty()) {
                    connection.setRequestProperty("Authorization", authorization);
                }

                statusCode = connection.getResponseCode();

                // Check Content-Type for captive portal detection
                String contentTypeHeader = connection.getContentType();
                if (statusCode >= 200 && statusCode < 300 && contentTypeHeader != null) {
                    String lowerContentType = contentTypeHeader.toLowerCase();
                    if (!lowerContentType.contains("application/json") &&
                        !lowerContentType.contains("text/json")) {
                        error = "Unexpected Content-Type: " + contentTypeHeader +
                                " (expected application/json). Possible captive portal or proxy interference.";
                        statusCode = -1;
                        nativeOnResponse(requestId, statusCode, null, error);
                        return;
                    }
                }

                BufferedReader reader = null;
                try {
                    if (statusCode >= 200 && statusCode < 300) {
                        reader = new BufferedReader(new InputStreamReader(connection.getInputStream(), StandardCharsets.UTF_8));
                    } else if (connection.getErrorStream() != null) {
                        reader = new BufferedReader(new InputStreamReader(connection.getErrorStream(), StandardCharsets.UTF_8));
                    }

                    if (reader != null) {
                        StringBuilder response = new StringBuilder();
                        String line;
                        int totalSize = 0;
                        while ((line = reader.readLine()) != null) {
                            totalSize += line.length();
                            if (totalSize > MAX_RESPONSE_SIZE) {
                                throw new IOException("Response body exceeds maximum size of " + MAX_RESPONSE_SIZE + " bytes");
                            }
                            response.append(line);
                        }
                        responseBody = response.toString();
                    }
                } finally {
                    if (reader != null) {
                        try { reader.close(); } catch (IOException ignored) {}
                    }
                }

                if (statusCode < 200 || statusCode >= 300) {
                    error = "HTTP " + statusCode + ": " + (responseBody != null ? responseBody : "Unknown error");
                }

            } catch (SocketTimeoutException e) {
                Log.e(TAG, "HTTP GET timed out: " + e.getMessage());
                statusCode = -1;
                error = "Request timed out: " + e.getMessage();
            } catch (UnknownHostException e) {
                Log.e(TAG, "Unknown host: " + e.getMessage());
                statusCode = -1;
                error = "Unknown host (DNS resolution failed): " + e.getMessage();
            } catch (SSLException e) {
                Log.e(TAG, "SSL error: " + e.getMessage());
                statusCode = -1;
                error = "SSL/TLS error: " + e.getMessage();
            } catch (IOException e) {
                Log.e(TAG, "HTTP GET failed: " + e.getMessage(), e);
                statusCode = -1;
                error = "Network error: " + e.getMessage();
            } catch (Exception e) {
                Log.e(TAG, "Unexpected error: " + e.getMessage(), e);
                statusCode = -1;
                error = "Error: " + e.getMessage();
            } finally {
                if (connection != null) {
                    try {
                        connection.disconnect();
                    } catch (Exception ignored) {}
                }
            }

            try {
                nativeOnResponse(requestId, statusCode, responseBody, error);
            } catch (Exception e) {
                Log.e(TAG, "Failed to deliver response to native: " + e.getMessage(), e);
            }
        });
    }

    /**
     * Shutdown the executor service.
     * Should be called when the application is terminating.
     */
    public static void shutdown() {
        executor.shutdown();
    }
}
