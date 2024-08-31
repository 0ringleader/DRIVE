// Author: Alina Krasnobaieva
// Date: 20.08.2024
// Description: This script sends the values detected by OffRoadDetection.cs to the third-party clients in JSON format

using System;
using System.Net;
using System.Text;
using System.Threading;
using UnityEngine;

public class RoadStatusSender : MonoBehaviour
{
    public int port = 8000;  // Port number to listen for HTTP requests
    private HttpListener httpListener;  // HTTP listener to handle client connections
    private bool isRunning;  // Boolean flag to control the server's running state
    private bool isOffRoad;  // Boolean flag to store the off-road status

    // Counter for times the car went off road as the boolean can't be checked fast enough by training clients
    private int failureCount = 0;

    void Start()
    {
        Application.runInBackground = true;  // Allow the application to run in the background
        StartServer();  // Start the HTTP server
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
                    var context = httpListener.GetContext();  // Get the context (incoming request)
                    ThreadPool.QueueUserWorkItem(o => HandleClient(context));  // Handle the client in a separate thread
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
        var response = context.Response;  // Get the response object
        response.ContentType = "application/json";  // Set the content type to JSON
        response.StatusCode = (int)HttpStatusCode.OK;  // Set the status code to 200 (OK)

        Debug.Log("Client connected.");

        try
        {
            // Create a JSON string representing the status and failure count
            string jsonResponse = $"{{ \"offRoad\": {isOffRoad.ToString().ToLower()}, \"failureCount\": {failureCount} }}";
            byte[] jsonBytes = Encoding.UTF8.GetBytes(jsonResponse);

            response.OutputStream.Write(jsonBytes, 0, jsonBytes.Length);  // Send the JSON response
            response.OutputStream.Flush();  // Ensure the data is sent immediately
            Debug.Log($"Status sent: {jsonResponse}");
        }
        catch (Exception ex)
        {
            Debug.LogError($"Error sending status: {ex.Message}");
        }
        finally
        {
            response.OutputStream.Close();  // Close the response stream
            Debug.Log("Client disconnected.");
        }
    }

    void OnApplicationQuit()
    {
        Debug.Log("Stopping server...");
        isRunning = false;
        if (httpListener != null && httpListener.IsListening)
        {
            httpListener.Stop();  // Stop the HTTP listener
            httpListener.Close();  // Close the HTTP listener
        }
    }
    
    // Public method to set the road status and increment the failure count if off-road
    public void setRoadStatus(bool offRoad)
    {
        isOffRoad = offRoad;
        if (offRoad)
        {
            failureCount++;  // Increment the failure count when the car goes off-road
        }
    }
}
