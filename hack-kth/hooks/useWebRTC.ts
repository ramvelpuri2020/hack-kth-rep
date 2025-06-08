import { useState, useRef, useCallback } from 'react';
import { db } from '../lib/firebase';
import { 
  collection, 
  doc, 
  addDoc, 
  getDoc, 
  setDoc, 
  updateDoc, 
  onSnapshot 
} from 'firebase/firestore';

const servers = {
  iceServers: [
    {
      urls: ["stun:stun1.l.google.com:19302", "stun:stun2.l.google.com:19302"],
    },
  ],
  iceCandidatePoolSize: 10,
};

export const useWebRTC = () => {
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);
  const [remoteStream, setRemoteStream] = useState<MediaStream | null>(null);
  const [roomId, setRoomId] = useState<string>('');
  const [isConnecting, setIsConnecting] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string>('');

  const pcRef = useRef<RTCPeerConnection | null>(null);
  const localVideoRef = useRef<HTMLVideoElement>(null);
  const remoteVideoRef = useRef<HTMLVideoElement>(null);

  const startWebcam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });
      setLocalStream(stream);
      
      if (localVideoRef.current) {
        localVideoRef.current.srcObject = stream;
      }

      // Initialize peer connection
      pcRef.current = new RTCPeerConnection(servers);
      
      // Add tracks to peer connection
      stream.getTracks().forEach((track) => {
        pcRef.current?.addTrack(track, stream);
      });

      // Setup remote stream
      const remoteStreamObj = new MediaStream();
      setRemoteStream(remoteStreamObj);

      pcRef.current.ontrack = (event) => {
        event.streams[0].getTracks().forEach((track) => {
          remoteStreamObj.addTrack(track);
        });
        if (remoteVideoRef.current) {
          remoteVideoRef.current.srcObject = remoteStreamObj;
        }
      };

      pcRef.current.onconnectionstatechange = () => {
        if (pcRef.current?.connectionState === 'connected') {
          setIsConnected(true);
          setIsConnecting(false);
        } else if (pcRef.current?.connectionState === 'failed') {
          setError('Connection failed');
          setIsConnecting(false);
        }
      };

    } catch (err) {
      setError('Failed to access webcam');
      console.error('Error accessing webcam:', err);
    }
  }, []);

  const createRoom = useCallback(async () => {
    if (!pcRef.current) {
      setError('Please start your webcam first');
      return;
    }

    try {
      setIsConnecting(true);
      setError('');

      const roomRef = doc(collection(db, 'rooms'));
      const offerCandidates = collection(roomRef, 'offerCandidates');
      const answerCandidates = collection(roomRef, 'answerCandidates');

      setRoomId(roomRef.id);

      // Collect ICE candidates
      pcRef.current.onicecandidate = (event) => {
        if (event.candidate) {
          addDoc(offerCandidates, event.candidate.toJSON());
        }
      };

      // Create offer
      const offerDescription = await pcRef.current.createOffer();
      await pcRef.current.setLocalDescription(offerDescription);

      const offer = {
        sdp: offerDescription.sdp,
        type: offerDescription.type,
      };

      await setDoc(roomRef, { offer });

      // Listen for remote answer
      onSnapshot(roomRef, (snapshot) => {
        const data = snapshot.data();
        if (!pcRef.current?.currentRemoteDescription && data?.answer) {
          const answerDescription = new RTCSessionDescription(data.answer);
          pcRef.current.setRemoteDescription(answerDescription);
        }
      });

      // Listen for remote ICE candidates
      onSnapshot(answerCandidates, (snapshot) => {
        snapshot.docChanges().forEach((change) => {
          if (change.type === 'added') {
            const candidate = new RTCIceCandidate(change.doc.data());
            pcRef.current?.addIceCandidate(candidate);
          }
        });
      });

    } catch (err) {
      setError('Failed to create room');
      setIsConnecting(false);
      console.error('Error creating room:', err);
    }
  }, []);

  const joinRoom = useCallback(async (roomIdToJoin: string) => {
    if (!pcRef.current) {
      setError('Please start your webcam first');
      return;
    }

    try {
      setIsConnecting(true);
      setError('');

      const roomRef = doc(db, 'rooms', roomIdToJoin);
      const roomSnapshot = await getDoc(roomRef);

      if (!roomSnapshot.exists()) {
        setError('Room not found');
        setIsConnecting(false);
        return;
      }

      const offerCandidates = collection(roomRef, 'offerCandidates');
      const answerCandidates = collection(roomRef, 'answerCandidates');

      // Collect ICE candidates
      pcRef.current.onicecandidate = (event) => {
        if (event.candidate) {
          addDoc(answerCandidates, event.candidate.toJSON());
        }
      };

      const roomData = roomSnapshot.data();
      const offerDescription = roomData?.offer;

      await pcRef.current.setRemoteDescription(new RTCSessionDescription(offerDescription));

      const answerDescription = await pcRef.current.createAnswer();
      await pcRef.current.setLocalDescription(answerDescription);

      const answer = {
        type: answerDescription.type,
        sdp: answerDescription.sdp,
      };

      await updateDoc(roomRef, { answer });

      // Listen for remote ICE candidates
      onSnapshot(offerCandidates, (snapshot) => {
        snapshot.docChanges().forEach((change) => {
          if (change.type === 'added') {
            const data = change.doc.data();
            pcRef.current?.addIceCandidate(new RTCIceCandidate(data));
          }
        });
      });

      setRoomId(roomIdToJoin);

    } catch (err) {
      setError('Failed to join room');
      setIsConnecting(false);
      console.error('Error joining room:', err);
    }
  }, []);

  const hangup = useCallback(() => {
    if (localStream) {
      localStream.getTracks().forEach(track => track.stop());
    }
    if (pcRef.current) {
      pcRef.current.close();
    }
    
    setLocalStream(null);
    setRemoteStream(null);
    setRoomId('');
    setIsConnecting(false);
    setIsConnected(false);
    setError('');
    
    pcRef.current = null;
  }, [localStream]);

  return {
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
  };
}; 