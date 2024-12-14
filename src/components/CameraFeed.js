'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Card } from '@/components/ui/card';

const CameraFeed = ({ onLetterDetected }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(document.createElement('canvas'));
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const enableCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Error accessing camera:", err);
        setError("Camera access failed");
      }
    };

    enableCamera();

    // Set up frame processing
    const processFrame = async () => {
      if (!videoRef.current || isProcessing) return;

      try {
        setIsProcessing(true);
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        // Make sure video is playing
        if (video.readyState !== video.HAVE_ENOUGH_DATA) return;

        // Match canvas size to video
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;

        // Draw the current frame
        ctx.drawImage(video, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg', 0.7);

        // Send to backend
        const response = await fetch('http://localhost:5000/detect', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ image: imageData })
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        if (data.error) {
          throw new Error(data.error);
        }

        if (data.letter) {
          onLetterDetected(data.letter);
          setError(null);
        }
      } catch (err) {
        console.error('Processing error:', err);
        setError("Server connection failed");
      } finally {
        setIsProcessing(false);
      }
    };

    // Process frames every 1000ms (reduced frequency for debugging)
    const interval = setInterval(processFrame, 1000);

    return () => {
      clearInterval(interval);
      const stream = videoRef.current?.srcObject;
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [onLetterDetected, isProcessing]);

  return (
    <Card>
      <div className="p-4">
        <div className="flex justify-between items-center">
          <h2 className="text-2xl font-semibold">Camera Feed</h2>
          {error && <span className="text-red-500 text-sm">{error}</span>}
        </div>
      </div>
      <div className="relative w-full h-[400px] bg-gray-100">
        <video 
          ref={videoRef}
          autoPlay 
          playsInline 
          muted
          className="absolute inset-0 w-full h-full object-cover"
        />
        {isProcessing && (
          <div className="absolute top-2 right-2 bg-blue-500 text-white px-3 py-1 rounded-full text-sm">
            Processing...
          </div>
        )}
      </div>
    </Card>
  );
};

export default CameraFeed;