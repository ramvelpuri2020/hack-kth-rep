'use client';

import { useState } from 'react';
import { useWebRTC } from '../hooks/useWebRTC';

export default function VideoCall() {
  const [joinRoomId, setJoinRoomId] = useState('');
  
  const {
    localStream,
    remoteStream,
    roomId,
    isConnecting,
    isConnected,
    error,
    localVideoRef,
    remoteVideoRef,
    startWebcam,
    createRoom,
    joinRoom,
    hangup,
  } = useWebRTC();

  const handleJoinRoom = () => {
    if (joinRoomId.trim()) {
      joinRoom(joinRoomId.trim());
    }
  };

  const copyRoomId = () => {
    navigator.clipboard.writeText(roomId);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-teal-400 to-teal-600 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-4">HandSync Video Call</h1>
        </div>

        {error && (
          <div className="bg-red-500 text-white p-4 rounded-lg mb-6 text-center">
            {error}
          </div>
        )}

        {/* Controls */}
        <div className="bg-white rounded-lg p-6 mb-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Start Webcam */}
            <div className="text-center">
              <h3 className="text-lg font-semibold mb-3">1. Start Your Camera</h3>
              <button
                onClick={startWebcam}
                disabled={localStream !== null}
                className={`px-6 py-3 rounded-lg font-medium transition-colors ${
                  localStream
                    ? 'bg-green-500 text-white cursor-not-allowed'
                    : 'bg-teal-500 hover:bg-teal-600 text-white'
                }`}
              >
                {localStream ? 'Camera Started' : 'Start Webcam'}
              </button>
            </div>

            {/* Create Room */}
            <div className="text-center">
              <h3 className="text-lg font-semibold mb-3">2. Create Room</h3>
              <button
                onClick={createRoom}
                disabled={!localStream || isConnecting || roomId !== ''}
                className={`px-6 py-3 rounded-lg font-medium transition-colors ${
                  !localStream || isConnecting || roomId
                    ? 'bg-gray-400 text-white cursor-not-allowed'
                    : 'bg-orange-500 hover:bg-orange-600 text-white'
                }`}
              >
                {isConnecting ? 'Creating...' : 'Create Room'}
              </button>
              {roomId && (
                <div className="mt-3">
                  <p className="text-sm text-gray-600 mb-2">Room ID:</p>
                  <div className="flex items-center justify-center gap-2">
                    <code className="bg-gray-100 px-3 py-1 rounded text-sm">
                      {roomId}
                    </code>
                    <button
                      onClick={copyRoomId}
                      className="bg-blue-500 hover:bg-blue-600 text-white px-3 py-1 rounded text-sm"
                    >
                      Copy
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Join Room */}
            <div className="text-center">
              <h3 className="text-lg font-semibold mb-3">3. Join Room</h3>
              <div className="space-y-2">
                <input
                  type="text"
                  placeholder="Enter Room ID"
                  value={joinRoomId}
                  onChange={(e) => setJoinRoomId(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                  disabled={!localStream || isConnecting}
                />
                <button
                  onClick={handleJoinRoom}
                  disabled={!localStream || !joinRoomId.trim() || isConnecting}
                  className={`w-full px-4 py-2 rounded-lg font-medium transition-colors ${
                    !localStream || !joinRoomId.trim() || isConnecting
                      ? 'bg-gray-400 text-white cursor-not-allowed'
                      : 'bg-purple-500 hover:bg-purple-600 text-white'
                  }`}
                >
                  {isConnecting ? 'Joining...' : 'Join Room'}
                </button>
              </div>
            </div>
          </div>

          {/* Connection Status */}
          {isConnected && (
            <div className="text-center mt-4">
              <span className="bg-green-100 text-green-800 px-4 py-2 rounded-lg">
                ✅ Connected
              </span>
            </div>
          )}
        </div>

        {/* Video Streams */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Local Video */}
          <div className="bg-white rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3 text-center">Your Video</h3>
            <div className="relative aspect-video bg-gray-900 rounded-lg overflow-hidden">
              <video
                ref={localVideoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
              />
              {!localStream && (
                <div className="absolute inset-0 flex items-center justify-center text-white">
                  <span>Start your webcam to see video</span>
                </div>
              )}
            </div>
          </div>

          {/* Remote Video */}
          <div className="bg-white rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3 text-center">Remote Video</h3>
            <div className="relative aspect-video bg-gray-900 rounded-lg overflow-hidden">
              <video
                ref={remoteVideoRef}
                autoPlay
                playsInline
                className="w-full h-full object-cover"
              />
              {!remoteStream && (
                <div className="absolute inset-0 flex items-center justify-center text-white">
                  <span>Waiting for remote connection...</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Hangup */}
        {(localStream || roomId) && (
          <div className="text-center">
            <button
              onClick={hangup}
              className="bg-red-500 hover:bg-red-600 text-white px-8 py-3 rounded-lg font-medium transition-colors"
            >
              Hangup
            </button>
          </div>
        )}

        {/* Back to Home */}
        <div className="text-center mt-6">
          <a
            href="/"
            className="text-white hover:text-gray-200 underline"
          >
            ← Back to Home
          </a>
        </div>
      </div>
    </div>
  );
} 