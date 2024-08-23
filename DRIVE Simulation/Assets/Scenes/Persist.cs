using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PersistentObject : MonoBehaviour
{
    private void Awake()
    {
        DontDestroyOnLoad(gameObject); // Verhindert, dass dieses Objekt bei Szenenwechseln gelöscht wird
    }
}

