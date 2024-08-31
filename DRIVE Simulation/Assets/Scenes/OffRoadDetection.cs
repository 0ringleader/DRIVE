// Author: Alina Krasnobaieva
// Date: 20.08.2024
// Description: This script monitors whether the vehicle leaves the road (goes off-road) and displays warning messages.
//              If the vehicle goes off-road, it can be reset or the scene can be switched if the corresponding options are enabled.

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;

public class OffRoadDetection : MonoBehaviour
{
    // Variables:
    // roadTag: Defines the tag that indicates whether the vehicle is on the road.
    public string roadTag = "RoadMesh";
    // wheelStatus: Dictionary that tracks the status of the four wheels (true = on the road, false = off-road).
    private Dictionary<string, bool> wheelStatus;
    // statusText: UI text field that displays warning messages when the vehicle is off-road.
    private Text statusText;
    // carController: Reference to the CustomCarController, which contains additional vehicle settings.
    private CustomCarController carController;
    // roadStatusSender: Reference to RoadStatusSender, which sends the off-road status.
    private RoadStatusSender roadStatusSender;

    void Start()
    {
        // Initializes variables and sets up UI elements.
        statusText = GameObject.Find("OffRoadWarning").GetComponent<Text>();
        carController = GameObject.Find("CarBody").GetComponent<CustomCarController>();
        roadStatusSender = GameObject.Find("CarBody").GetComponent<RoadStatusSender>();
        
        // The warning text field is initially deactivated.
        statusText.gameObject.SetActive(false);
        
        // All wheels are assumed to be on the road at the start.
        wheelStatus = new Dictionary<string, bool>
        {
            { "WheelHL", true },
            { "WheelHR", true },
            { "WheelVL", true },
            { "WheelVR", true }
        };

        // Update initial off-road status.
        UpdateOffRoadStatus();
    }

    void OnCollisionEnter(Collision collision)
    {
        // This method is called when one of the wheels comes into contact with a collider.
        Debug.Log("OnCollisionEnter aufgerufen mit Tag: " + collision.collider.tag);
        if (wheelStatus.ContainsKey(collision.collider.tag) || collision.collider.GetComponent<MeshCollider>() != null)
        {
            // The status of the corresponding wheel is set to "true" if it collides with the road.
            wheelStatus[collision.collider.tag] = true;
            // The off-road status is then updated.
            UpdateOffRoadStatus();
        }
    }

    void OnCollisionExit(Collision collision)
    {
        // This method is called when one of the wheels loses contact with a collider.
        Debug.Log("OnCollisionExit aufgerufen mit Tag: " + collision.collider.tag);
        if (wheelStatus.ContainsKey(collision.collider.tag) || collision.collider.GetComponent<MeshCollider>() != null)
        {
            // The status of the corresponding wheel is set to "false".
            wheelStatus[collision.collider.tag] = false;
            // The off-road status is then updated.
            UpdateOffRoadStatus();
        }
    }

    void UpdateOffRoadStatus()
    {
        // Checks if all wheels are either on the road or off-road.
        bool allOnRoad = true;
        bool allOffRoad = true;
        
        foreach (var status in wheelStatus.Values)
        {
            if (!status)
            {
                allOnRoad = false;
            }
            if (status)
            {
                allOffRoad = false;
            }
        }

        // Updates the off-road status based on the status values of the wheels.
        if (allOnRoad)
        {
            SetOffRoadStatus(false, null);
        }
        else if (allOffRoad)
        {
            SetOffRoadStatus(true, null);
        }
        else
        {
            foreach (var wheel in wheelStatus)
            {
                if (!wheel.Value)
                {
                    // Displays a warning if the vehicle is off-road.
                    SetOffRoadStatus(true, wheel.Key);
                    return;
                }
            }
        }
    }

    void SetOffRoadStatus(bool offRoad, string wheelTag)
    {
        // Sets the off-road status and displays corresponding warning messages in the UI.
        roadStatusSender.setRoadStatus(offRoad);
        Debug.Log("SetOffRoadStatus aufgerufen: " + offRoad);
        
        statusText.gameObject.SetActive(true);
        
        if (offRoad)
        {
            if (wheelTag == null)
            {
                statusText.text = "The car is not on the road.";
                statusText.color = Color.red;
                Debug.Log("The car is not on the road.");
            }
            else
            {
                statusText.text = $"The wheel {wheelTag} has left the road!";
                statusText.color = Color.red;
                Debug.Log($"The wheel {wheelTag} has left the road!");
            }

            // If the vehicle is off-road and the reset option is enabled,
            // the vehicle is reset, and the scene may be switched.
            if (carController != null)
            {
                if (carController.resetOnOffRoad)
                {
                    carController.ResetCar();
                    if (carController.autoSwitchScenes)
                    {
                        SwitchToRandomTrack();
                    }
                }
            }
        }
        else
        {
            statusText.text = "The car is on the road.";
            statusText.color = new Color(0.0f, 0.5f, 0.0f);
            Debug.Log("The car is on the road.");
        }
    }

    void SwitchToRandomTrack()
    {
        // Switches the scene to a random track if the option is enabled.
        GameObject.Find("Canvas").GetComponent<UIButtonHandler>().SwitchToRandomScene(); // Switch to random scene on track();
    }
}