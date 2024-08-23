using UnityEngine;
using UnityEngine.UI;
using TMPro;
using UnityEngine.SceneManagement; // Notwendig für Szenenmanagement
using System.Collections.Generic;

public class UIButtonHandler : MonoBehaviour
{
    public Material groundMaterial;
    public Texture groundTexture;
    public CustomCarController carController;
    public MJPEGStream mjpegStream;
    
    // UI-Elemente als Instanzvariablen
    private TextMeshProUGUI fpsText;
    private TextMeshProUGUI mouseWheelText;
    private TextMeshProUGUI speedText;
    private TextMeshProUGUI steeringText;
    private Slider gameSpeedSlider;
    private TextMeshProUGUI gameSpeedText;
    private TMP_Dropdown resolutionDropdown;
    private Toggle autoSceneSwitchToggle;
    private Button randomSceneButton;
    private Toggle toggleTextureToggle;
    private Button exitButton;
    private Button resetCarButton;
    private Toggle controlByWebsiteToggle;
    private Toggle toggleOffRoadReset;
    private Toggle syncToGameSpeedToggle;
    private TextMeshProUGUI offRoadWarning;
    
    private float deltaTime = 0.0f;
    private List<string> trackScenes = new List<string>
    {
        "Track1",
        "Track2",
        "Track3",
        "Track4",
        "Track5",
        "Track6",
        "Track7",
        "Track8",
        "Track9"
    };

    private bool isUIVisible = true; // Variable to track UI visibility

        void Start()
    {
        // Find the UI elements in the scene
        mouseWheelText = GameObject.Find("MousewheelText").GetComponent<TextMeshProUGUI>();
        speedText = GameObject.Find("SpeedText").GetComponent<TextMeshProUGUI>();
        steeringText = GameObject.Find("SteeringText").GetComponent<TextMeshProUGUI>();
        gameSpeedSlider = GameObject.Find("GameSpeedSlider").GetComponent<Slider>();
        gameSpeedText = GameObject.Find("GameSpeedText").GetComponent<TextMeshProUGUI>();
        resolutionDropdown = GameObject.Find("ResolutionDropdown").GetComponent<TMP_Dropdown>();
        fpsText = GameObject.Find("fpsText").GetComponent<TextMeshProUGUI>();
        autoSceneSwitchToggle = GameObject.Find("AutoSceneSwitchToggle").GetComponent<Toggle>();
        randomSceneButton = GameObject.Find("RandomSceneButton").GetComponent<Button>();

        toggleTextureToggle = GameObject.Find("ToggleTextureToggle").GetComponent<Toggle>();
        exitButton = GameObject.Find("ExitButton").GetComponent<Button>();
        resetCarButton = GameObject.Find("ResetCarButton").GetComponent<Button>();
        controlByWebsiteToggle = GameObject.Find("ControlByWebsiteToggle").GetComponent<Toggle>();
        toggleOffRoadReset = GameObject.Find("ToggleOffRoadReset").GetComponent<Toggle>();
        syncToGameSpeedToggle = GameObject.Find("SyncToGameSpeedToggle").GetComponent<Toggle>();
        offRoadWarning = GameObject.Find("OffRoadWarning").GetComponent<TextMeshProUGUI>();

        // Add listeners to the toggles and buttons
        toggleTextureToggle.onValueChanged.AddListener(delegate { ToggleGroundTexture(toggleTextureToggle); });
        exitButton.onClick.AddListener(ExitApplication);
        resetCarButton.onClick.AddListener(ResetCar);
        controlByWebsiteToggle.onValueChanged.AddListener(delegate { ToggleControlByWebsite(controlByWebsiteToggle); });
        toggleOffRoadReset.onValueChanged.AddListener(delegate { ToggleOffRoadReset(toggleOffRoadReset); });
        syncToGameSpeedToggle.onValueChanged.AddListener(delegate { ToggleSyncToGameSpeed(syncToGameSpeedToggle); });
        gameSpeedSlider.onValueChanged.AddListener(UpdateGameSpeed);
        resolutionDropdown.onValueChanged.AddListener(delegate { ChangeResolution(resolutionDropdown); });
        randomSceneButton.onClick.AddListener(SwitchToRandomScene); // Methode für den Random-Szenenwechsel hinzufügen
        autoSceneSwitchToggle.onValueChanged.AddListener(delegate { ToggleAutoSceneSwitch(autoSceneSwitchToggle); }); // Methode für Auto-Szenenwechsel hinzufügen

        PopulateResolutionDropdown();
        ToggleMouseWheelText(controlByWebsiteToggle.isOn);
        UpdateGameSpeed(gameSpeedSlider.value);
    }

    void Update()
    {
        // Update FPS
        deltaTime += (Time.unscaledDeltaTime - deltaTime) * 0.1f;

        int fps = Mathf.RoundToInt(1.0f / deltaTime);

        if (fpsText != null)
        {
            fpsText.text = $"FPS: {fps}";
        }

        if (Input.GetKeyDown(KeyCode.Escape))
        {
            ExitApplication();
        }

        if (Input.GetKeyDown(KeyCode.H))
        {
            ToggleUIVisibility();
        }
    }

    void ToggleUIVisibility()
    {
        isUIVisible = !isUIVisible;

        // Update the visibility of all UI elements
        mouseWheelText.gameObject.SetActive(isUIVisible);
        speedText.gameObject.SetActive(isUIVisible);
        steeringText.gameObject.SetActive(isUIVisible);
        gameSpeedSlider.gameObject.SetActive(isUIVisible);
        gameSpeedText.gameObject.SetActive(isUIVisible);
        resolutionDropdown.gameObject.SetActive(isUIVisible);
        fpsText.gameObject.SetActive(isUIVisible);
        autoSceneSwitchToggle.gameObject.SetActive(isUIVisible);
        randomSceneButton.gameObject.SetActive(isUIVisible);
        toggleTextureToggle.gameObject.SetActive(isUIVisible);
        exitButton.gameObject.SetActive(isUIVisible);
        resetCarButton.gameObject.SetActive(isUIVisible);
        controlByWebsiteToggle.gameObject.SetActive(isUIVisible);
        toggleOffRoadReset.gameObject.SetActive(isUIVisible);
        syncToGameSpeedToggle.gameObject.SetActive(isUIVisible);
        offRoadWarning.gameObject.SetActive(isUIVisible);
    }

    void PopulateResolutionDropdown()
    {
        resolutionDropdown.ClearOptions();
        Resolution[] resolutions = Screen.resolutions;
        List<string> options = new List<string>();

        foreach (Resolution res in resolutions)
        {
            string option = res.width + " x " + res.height;
            options.Add(option);
        }

        resolutionDropdown.AddOptions(options);
    }

    void SetResolution(int resolutionIndex)
    {
        Resolution resolution = Screen.resolutions[resolutionIndex];
        Screen.SetResolution(resolution.width, resolution.height, Screen.fullScreen);
    }

    void ToggleGroundTexture(Toggle toggle)
    {
        if (toggle.isOn)
        {
            groundMaterial.SetTexture("_MainTex", groundTexture);
        }
        else
        {
            groundMaterial.SetTexture("_MainTex", null);
        }
    }

    void ExitApplication()
    {
        Application.Quit();
    }

    void ResetCar()
    {
        if (carController != null)
        {
            carController.ResetCar();
        }
    }

    void ToggleControlByWebsite(Toggle toggle)
    {
        if (carController != null)
        {
            carController.isControlledByWebsite = toggle.isOn;
        }
        ToggleMouseWheelText(toggle.isOn);
    }

    void ToggleMouseWheelText(bool isControlledByWebsite)
    {
        if (mouseWheelText != null)
        {
            mouseWheelText.text = isControlledByWebsite ? "" : "Control speed with mousewheel or arrow keys";
        }
    }

    void ToggleOffRoadReset(Toggle toggle)
    {
        if (carController != null)
        {
            carController.resetOnOffRoad = toggle.isOn;
        }
    }

    void ToggleSyncToGameSpeed(Toggle toggle)
    {
        if (mjpegStream != null)
        {
            mjpegStream.syncToGameSpeed = toggle.isOn;
        }
    }

    void ToggleAutoSceneSwitch(Toggle toggle)
    {
        carController.autoSwitchScenes = toggle.isOn; // Den Status des Auto-Switch-Toggles speichern
    }

    public void UpdateSpeedText(float speed, float mappedSpeed)
    {
        if (speedText != null)
        {
            speedText.text = $"Speed: {speed:F2}m/s, value: {mappedSpeed:F0}";
        }
    }

    public void UpdateSteeringText(float steering, float mappedSteering)
    {
        if (steeringText != null)
        {
            steeringText.text = $"Steering: {steering:F2}°, value: {mappedSteering:F0}";
        }
    }

    void UpdateGameSpeed(float newSpeed)
    {
        newSpeed = newSpeed / 10f;
        Time.timeScale = newSpeed;

        if (gameSpeedText != null)
        {
            gameSpeedText.text = $"Game Speed: {newSpeed:F2}x";
        }
        if (mjpegStream != null)
        {
            mjpegStream.setGameSpeed(newSpeed);
        }
    }

    void ChangeResolution(TMP_Dropdown dropdown)
    {
        SetResolution(dropdown.value);
    }

    public void SwitchToRandomScene()
    {
        // Falls keine Track-Szenen vorhanden sind
        if (trackScenes.Count == 0)
        {
            Debug.LogWarning("Keine Track-Szenen gefunden.");
            return;
        }

        carController.ResetCar();
        // Zufällige Szene auswählen
        int randomIndex = Random.Range(0, trackScenes.Count);
        SceneManager.LoadScene(trackScenes[randomIndex]);
    }
}