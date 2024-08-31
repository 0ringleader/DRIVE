using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;

public class OffRoadDetection : MonoBehaviour
{
    public string roadTag = "RoadMesh";
    private Dictionary<string, bool> wheelStatus;
    private Text statusText;
    private CustomCarController carController;
    private RoadStatusSender roadStatusSender;
    void Start()
    {
        statusText = GameObject.Find("OffRoadWarning").GetComponent<Text>();
        carController = GameObject.Find("CarBody").GetComponent<CustomCarController>();
        roadStatusSender = GameObject.Find("CarBody").GetComponent<RoadStatusSender>();
        wheelStatus = new Dictionary<string, bool>
        {
            { "WheelHL", true },
            { "WheelHR", true },
            { "WheelVL", true },
            { "WheelVR", true }
        };
        statusText.gameObject.SetActive(false);
        UpdateOffRoadStatus();
    }

    void OnCollisionEnter(Collision collision)
    {
        Debug.Log("OnCollisionEnter aufgerufen mit Tag: " + collision.collider.tag);
        if (wheelStatus.ContainsKey(collision.collider.tag) || collision.collider.GetComponent<MeshCollider>() != null)
        {
            wheelStatus[collision.collider.tag] = true;
            UpdateOffRoadStatus();
        }
    }

    void OnCollisionExit(Collision collision)
    {
        Debug.Log("OnCollisionExit aufgerufen mit Tag: " + collision.collider.tag);
        if (wheelStatus.ContainsKey(collision.collider.tag) || collision.collider.GetComponent<MeshCollider>() != null)
        {
            wheelStatus[collision.collider.tag] = false;
            UpdateOffRoadStatus();
        }
    }

    void UpdateOffRoadStatus()
    {
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
                    SetOffRoadStatus(true, wheel.Key);
                    return;
                }
            }
        }
    }

    void SetOffRoadStatus(bool offRoad, string wheelTag)
    {
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

            // Reset the car if resetOnOffRoad is true
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
        GameObject.Find("Canvas").GetComponent<UIButtonHandler>().SwitchToRandomScene(); // Switch to random scene on track();
    }
    
}
