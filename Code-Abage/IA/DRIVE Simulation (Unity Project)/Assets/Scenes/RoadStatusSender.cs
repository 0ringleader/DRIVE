using System;
using System.Net;
using System.Text;
using System.Threading;
using UnityEngine;

public class RoadStatusSender : MonoBehaviour
{
    public int port = 8000;
    private HttpListener httpListener;
    private bool isRunning;
    private bool isOffRoad;

    // Counter for times the car went off road as the boolean cant be checked fast enough by training clients
    private int failureCount = 0;

    void Start()
    {
        Application.runInBackground = true;
        StartServer();
    }

    void StartServer()
    {
        Debug.Log("Starting server...");
        httpListener = new HttpListener();
        httpListener.Prefixes.Add($"http://*:{port}/roadStatus/");
        httpListener.Start();
        isRunning = true;

        // Run the HttpListener in a separate thread
        new Thread(() =>
        {
            while (isRunning)
            {
                Debug.Log("Waiting for client connection...");
                try
                {
                    var context = httpListener.GetContext();
                    ThreadPool.QueueUserWorkItem(o => HandleClient(context));
                }
                catch (Exception e)
                {
                    Debug.LogError($"Error accepting clients: {e.Message}");
                }
            }
        }).Start();

        Debug.Log($"Road status available at http://localhost:{port}/roadStatus/");
    }

    void HandleClient(HttpListenerContext context)
    {
        var response = context.Response;
        response.ContentType = "application/json";
        response.StatusCode = (int)HttpStatusCode.OK;

        Debug.Log("Client connected.");

        try
        {

            // Create a JSON string representing the status and failure count
            string jsonResponse = $"{{ \"offRoad\": {isOffRoad.ToString().ToLower()}, \"failureCount\": {failureCount} }}";
            byte[] jsonBytes = Encoding.UTF8.GetBytes(jsonResponse);

            response.OutputStream.Write(jsonBytes, 0, jsonBytes.Length);
            response.OutputStream.Flush();
            Debug.Log($"Status sent: {jsonResponse}");
        }
        catch (Exception ex)
        {
            Debug.LogError($"Error sending status: {ex.Message}");
        }
        finally
        {
            response.OutputStream.Close();
            Debug.Log("Client disconnected.");
        }
    }

    void OnApplicationQuit()
    {
        Debug.Log("Stopping server...");
        isRunning = false;
        if (httpListener != null && httpListener.IsListening)
        {
            httpListener.Stop();
            httpListener.Close();
        }
    }
    
    public void setRoadStatus(bool offRoad)
    {
        isOffRoad = offRoad;
        if (offRoad)
        {
            failureCount++;
        }
    }
}
