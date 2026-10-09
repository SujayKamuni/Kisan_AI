package com.kisanai.app;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.view.WindowManager;
import android.webkit.GeolocationPermissions;
import android.webkit.PermissionRequest;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.content.SharedPreferences;
import android.text.InputType;
import android.widget.EditText;
import androidx.appcompat.app.AlertDialog;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.view.WindowCompat;

public class MainActivity extends AppCompatActivity {

    // Preferences for saving the URL
    private static final String PREFS_NAME = "KisanAIPrefs";
    private static final String PREF_URL = "server_url";
    private String currentUrl;

    private WebView webView;
    private PermissionRequest pendingPermRequest;
    private GeolocationPermissions.Callback pendingGeolocationCallback;
    private String pendingGeolocationOrigin;

    // Runtime permission launchers
    private final ActivityResultLauncher<String[]> multiPermLauncher =
        registerForActivityResult(new ActivityResultContracts.RequestMultiplePermissions(), grants -> {
            boolean micOk = Boolean.TRUE.equals(grants.get(Manifest.permission.RECORD_AUDIO));
            if (pendingPermRequest != null) {
                if (micOk) {
                    pendingPermRequest.grant(pendingPermRequest.getResources());
                } else {
                    pendingPermRequest.deny();
                    Toast.makeText(this, "Microphone permission required for voice input", Toast.LENGTH_LONG).show();
                }
                pendingPermRequest = null;
            }
        });

    private final ActivityResultLauncher<String[]> locationLauncher =
        registerForActivityResult(new ActivityResultContracts.RequestMultiplePermissions(), grants -> {
            boolean locOk =
                Boolean.TRUE.equals(grants.get(Manifest.permission.ACCESS_FINE_LOCATION)) ||
                Boolean.TRUE.equals(grants.get(Manifest.permission.ACCESS_COARSE_LOCATION));
            if (pendingGeolocationCallback != null) {
                pendingGeolocationCallback.invoke(pendingGeolocationOrigin, locOk, false);
                pendingGeolocationCallback = null;
            }
        });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Edge-to-edge display — content draws behind status/navigation bars
        WindowCompat.setDecorFitsSystemWindows(getWindow(), false);
        getWindow().setStatusBarColor(Color.TRANSPARENT);
        getWindow().setNavigationBarColor(Color.TRANSPARENT);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            getWindow().getAttributes().layoutInDisplayCutoutMode =
                WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES;
        }

        setContentView(R.layout.activity_main);

        webView = findViewById(R.id.webView);
        setupWebView();
        
        // Show URL dialog on startup
        showUrlInputDialog();
    }

    private void showUrlInputDialog() {
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        String savedUrl = prefs.getString(PREF_URL, "https://blepharitic-flawiest-keenan.ngrok-free.dev/mobile");

        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Enter Server URL");
        builder.setCancelable(false); // Force user to provide a URL or exit

        final EditText input = new EditText(this);
        input.setInputType(InputType.TYPE_TEXT_VARIATION_URI);
        input.setText(savedUrl);
        builder.setView(input);

        builder.setPositiveButton("Connect", (dialog, which) -> {
            currentUrl = input.getText().toString().trim();
            if (!currentUrl.isEmpty()) {
                prefs.edit().putString(PREF_URL, currentUrl).apply();
                webView.loadUrl(currentUrl);
            } else {
                Toast.makeText(this, "URL cannot be empty", Toast.LENGTH_SHORT).show();
                showUrlInputDialog();
            }
        });

        builder.setNegativeButton("Exit", (dialog, which) -> finish());

        builder.show();
    }

    private void setupWebView() {
        WebSettings settings = webView.getSettings();

        // Core
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);       // localStorage
        settings.setDatabaseEnabled(true);

        // Media
        settings.setMediaPlaybackRequiresUserGesture(false); // auto-play audio
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_ALWAYS_ALLOW); // http+https

        // UX
        settings.setLoadWithOverviewMode(true);
        settings.setUseWideViewPort(true);
        settings.setSupportZoom(false);
        settings.setBuiltInZoomControls(false);
        settings.setDisplayZoomControls(false);

        // Geolocation
        settings.setGeolocationEnabled(true);

        // Debugging (disable in release)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
            WebView.setWebContentsDebuggingEnabled(true);
        }

        // WebViewClient — keeps navigation inside the WebView
        webView.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                // Stay in-app for the same host; open external links in browser
                Uri uri = request.getUrl();
                if (currentUrl == null) return false;
                
                String serverHost = Uri.parse(currentUrl).getHost();
                if (serverHost != null && serverHost.equals(uri.getHost())) {
                    return false; // load in WebView
                }
                // External URL — let Android handle it
                return true;
            }
        });

        // WebChromeClient — handles JS permissions
        webView.setWebChromeClient(new WebChromeClient() {

            // Microphone (and camera) permission from JS
            @Override
            public void onPermissionRequest(PermissionRequest request) {
                // Check if RECORD_AUDIO is already granted
                boolean micGranted = ContextCompat.checkSelfPermission(
                    MainActivity.this, Manifest.permission.RECORD_AUDIO)
                    == PackageManager.PERMISSION_GRANTED;

                if (micGranted) {
                    request.grant(request.getResources());
                } else {
                    pendingPermRequest = request;
                    multiPermLauncher.launch(new String[]{
                        Manifest.permission.RECORD_AUDIO
                    });
                }
            }

            // Geolocation permission from JS
            @Override
            public void onGeolocationPermissionsShowPrompt(
                    String origin, GeolocationPermissions.Callback callback) {
                boolean fineGranted = ContextCompat.checkSelfPermission(
                    MainActivity.this, Manifest.permission.ACCESS_FINE_LOCATION)
                    == PackageManager.PERMISSION_GRANTED;
                boolean coarseGranted = ContextCompat.checkSelfPermission(
                    MainActivity.this, Manifest.permission.ACCESS_COARSE_LOCATION)
                    == PackageManager.PERMISSION_GRANTED;

                if (fineGranted || coarseGranted) {
                    callback.invoke(origin, true, false);
                } else {
                    pendingGeolocationCallback = callback;
                    pendingGeolocationOrigin   = origin;
                    locationLauncher.launch(new String[]{
                        Manifest.permission.ACCESS_FINE_LOCATION,
                        Manifest.permission.ACCESS_COARSE_LOCATION
                    });
                }
            }
        });
    }

    // Back button — navigate WebView history before exiting
    @Override
    public void onBackPressed() {
        if (webView.canGoBack()) {
            webView.goBack();
        } else {
            super.onBackPressed();
        }
    }
}
