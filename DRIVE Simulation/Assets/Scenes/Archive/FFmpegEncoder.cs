using System.Diagnostics;
using UnityEngine;
using Debug = UnityEngine.Debug;

public class FFmpegEncoder : MonoBehaviour
{
    private Process ffmpegProcess;

    void Start()
    {
        StartEncoding();
    }

    public void StartEncoding()
    {
        // Set up the FFmpeg process
        ffmpegProcess = new Process();
        ffmpegProcess.StartInfo.FileName = "ffmpeg"; // Path to FFmpeg executable
        ffmpegProcess.StartInfo.Arguments = "-f mjpeg -i pipe:0 -c:v copy -f mjpeg http://localhost:1234/stream.mjpeg"; // Input and output format
        ffmpegProcess.StartInfo.UseShellExecute = false;
        ffmpegProcess.StartInfo.RedirectStandardInput = true;

        // Start the FFmpeg process
        ffmpegProcess.Start();
    }

    public void StopEncoding()
    {
        // Close the input stream, which will cause FFmpeg to exit
        ffmpegProcess.StandardInput.Close();
        ffmpegProcess.WaitForExit();
    }

    public void Update()
    {
        // Capture the screen as a Texture2D
        Texture2D screenshot = ScreenCapture.CaptureScreenshotAsTexture();

        if (screenshot == null)
        {
            Debug.LogError("Failed to capture screenshot");
            return;
        }

        // Convert the screenshot to a JPEG image
        byte[] jpegData = screenshot.EncodeToJPG();

        if (jpegData == null || jpegData.Length == 0)
        {
            Debug.LogError("Failed to encode screenshot to JPEG");
            return;
        }

        // Check if ffmpegProcess and its StandardInput and BaseStream are not null
        if (ffmpegProcess == null || ffmpegProcess.StandardInput == null || ffmpegProcess.StandardInput.BaseStream == null)
        {
            Debug.LogError("FFmpeg process or its input stream is not initialized");
            return;
        }

        // Write the JPEG data to FFmpeg's standard input
        ffmpegProcess.StandardInput.BaseStream.Write(jpegData, 0, jpegData.Length);

        // Clean up the screenshot Texture2D
        Object.Destroy(screenshot);
    }
}