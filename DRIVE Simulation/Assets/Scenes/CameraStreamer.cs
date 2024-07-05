using UnityEngine;
using System.Collections;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System;
using System.IO;

public class MjpegStream : MonoBehaviour
{
    public Camera cameraToUse;
    private UdpClient udpClient;
    private RenderTexture renderTexture;
    private Thread sendThread;

    void Start()
    {
        // Create a RenderTexture and assign it to the camera
        renderTexture = new RenderTexture(1280, 720, 24);
        cameraToUse.targetTexture = renderTexture;

        // Start the UDP client
        udpClient = new UdpClient();
        udpClient.Connect("127.0.0.1", 8888); // Replace with your server IP and port

        // Start a thread to send the video stream
        sendThread = new Thread(SendVideoStream);
        sendThread.Start();
    }

    void OnDestroy()
    {
        // Stop the thread when the game object is destroyed
        sendThread.Abort();
        udpClient.Close();
    }

    void SendVideoStream()
    {
        while (true)
        {
            // Capture the image from the RenderTexture
            RenderTexture.active = renderTexture;
            Texture2D texture = new Texture2D(renderTexture.width, renderTexture.height);
            texture.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
            texture.Apply();
            RenderTexture.active = null;

            // Convert the image to a JPEG format
            byte[] jpegData = ImageConversion.EncodeToJPG(texture);
        
            string documentsPath = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
            string filePath = Path.Combine(documentsPath, "test.jpg");
            File.WriteAllBytes(filePath, jpegData);
            
            // Check if the JPEG data is correctly generated
            if (jpegData == null || jpegData.Length == 0)
            {
                Debug.LogError("Failed to generate JPEG data");
                continue;
            }

            // Send the JPEG image over the UDP network stream
            try
            {
                udpClient.Send(jpegData, jpegData.Length);
                
            }
            catch (Exception e)
            {
                Debug.LogError("Failed to send data: " + e.Message);
            }

            Thread.Sleep(33); // Sleep for a short time to control the frame rate
        }
    }
}