'use client';

import { useState } from 'react';
import CameraFeed from '@/components/CameraFeed';
import { Card } from '@/components/ui/card';

export default function Home() {
  const [currentLetter, setCurrentLetter] = useState('None');
  const [currentWord, setCurrentWord] = useState('');

  const handleLetterDetected = (letter) => {
    setCurrentLetter(letter);
    // Add additional logic here for word building
  };

  return (
    <div className="min-h-screen p-8">
      <h1 className="text-4xl font-bold mb-8">ASL Word Prediction</h1>
      
      <div className="max-w-4xl mx-auto space-y-6">
        <Card className="p-4">
          <h2 className="text-2xl">Current Letter: <span className="font-bold">{currentLetter}</span></h2>
        </Card>

        <CameraFeed onLetterDetected={handleLetterDetected} />

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card className="p-4">
            <h2 className="text-2xl mb-4">Current Word</h2>
            <div className="text-lg font-mono">{currentWord || '_ _ _ _'}</div>
          </Card>

          <Card className="p-4">
            <h2 className="text-2xl mb-4">Suggestions</h2>
            <div className="text-gray-500">No suggestions available</div>
          </Card>
        </div>
      </div>
    </div>
  );
}