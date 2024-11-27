"use client";

import React, { useRef, useEffect, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { Input } from "./components/ui/input";
import { Button } from "./components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./components/ui/card";
import { Send, ChevronRight, ChevronLeft } from "lucide-react";
import ColorThief from "colorthief";
// import { rgb, hsl, wcagLuminance, wcagContrast } from "culori";

import sampleDataMap from "./data";

let sample_data = sampleDataMap["ds"];

import teamsData from "./teamsData.json";

const team2color = (team: string): number => {
  const teamData = teamsData.find((t) => t.fullName === team);
  if (teamData) {
    return hexstring2number(teamData.color);
  }
  return 0x000000;
};

type Vector3 = [number, number, number];

const player_images_dir = "/player_images/";
const team_images_dir = "/team_images/";

const createPlayersFromData = (
  data: Player[] = sample_data as unknown as Player[],
) => {
  // Console log every team name once
  let teams = new Set();
  Object.values(data).forEach((player) => {
    teams.add(player.team);
  });
  // console.log(teams);

  let players_tmp = Object.values(data);
  let center = true;
  if (center) {
    // Center the embeddings to the origin
    let emb_0_sum = 0;
    let emb_1_sum = 0;
    let emb_2_sum = 0;
    players_tmp.forEach((player) => {
      emb_0_sum += player.emb_0;
      emb_1_sum += player.emb_1;
      emb_2_sum += player.emb_2;
    });
    let emb_0_mean = emb_0_sum / players_tmp.length;
    let emb_1_mean = emb_1_sum / players_tmp.length;
    let emb_2_mean = emb_2_sum / players_tmp.length;

    players_tmp.forEach((player) => {
      player.emb_0 -= emb_0_mean;
      player.emb_1 -= emb_1_mean;
      player.emb_2 -= emb_2_mean;
    });
  }

  return players_tmp;
};

type Player = {
  id: string;
  name: string;
  position: string;
  team: string;
  emb_0: number;
  emb_1: number;
  emb_2: number;
  // otras propiedades de Player
};

interface DotData {
  id: number;
  title: string;
  content: JSX.Element;
  position: THREE.Vector3;
  mesh: THREE.Mesh;
  explored: boolean;
  topic: Player | any;
  color: number;
  team: string;
}
const hexstring2number = (hex: string): number => {
  return parseInt(hex.replace(/^#/, ""), 16);
};

const Dashboard = () => {
  const mountRef = useRef<HTMLDivElement>(null);
  const [selectedDotData, setSelectedDotData] = useState<DotData | null>(null);
  const [question, setQuestion] = useState("");
  const [dots, setDots] = useState<DotData[]>([]);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const highlightMeshRef = useRef<THREE.Mesh | null>(null);
  const [isSearchMode, setIsSearchMode] = useState(false);
  const [isLearnMode, setIsLearnMode] = useState(false);

  const [toggleColor, setToggleColor] = useState(false);

  const dotsRef = useRef(dots);

  const [data, setData] = useState<Player[]>(
    sample_data as unknown as Player[],
  );
  const [players, setPlayers] = useState<Player[]>(createPlayersFromData(data));

  useEffect(() => {
    setPlayers(createPlayersFromData(data));
    setSelectedDotData(null);
    console.log("Players:", players);
  }, [data]);

  useEffect(() => {
    dotsRef.current = dots;
  }, [dots]);

  useEffect(() => {
    if (!mountRef.current) return;

    const scene = new THREE.Scene();
    sceneRef.current = scene;
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000,
    );
    cameraRef.current = camera;
    const renderer = new THREE.WebGLRenderer({ antialias: true });

    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0xffffff);
    mountRef.current.appendChild(renderer.domElement);

    const ambientLight = new THREE.AmbientLight(0xffffff, 1);
    scene.add(ambientLight);
    const pointLight = new THREE.PointLight(0xffffff, 1);
    pointLight.position.set(5, 5, 5);
    scene.add(pointLight);

    const geometry = new THREE.SphereGeometry(0.03, 32, 32);
    const initialMaterial = new THREE.MeshStandardMaterial({ color: 0x000000 });

    const img2color = (img: string): Promise<number> => {
      return new Promise((resolve, reject) => {
        if (img === "") {
          resolve(0x000000);
          return;
        }
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        const image = new Image();
        image.crossOrigin = "Anonymous"; // Permitir solicitudes CORS
        image.src = img;
        image.onload = () => {
          try {
            ctx!.drawImage(image, 0, 0, canvas.width, canvas.height);
            const colorThief = new ColorThief();
            const pallete = colorThief.getPalette(image, 5);
            const maincolor = pallete[0];

            let [r, g, b] = maincolor;
            if (r && g && b) {
              resolve((r << 16) + (g << 8) + b);
            }
            resolve(0x000000);
          } catch (e) {
            console.error("Error al obtener los datos de la imagen:", e);
            resolve(0x000000);
          }
        };
        image.onerror = (e) => {
          console.error("Error al cargar la imagen:", e);
          resolve(0x000000);
        };
      });
    };

    Promise.all(
      players.map(async (player) => {
        const player_image =
          player_images_dir + "p" + player.id + "_250x250.png";
        const team = teamsData.find((team) => team.fullName === player.team);
        const team_image = team?.web_image
          ? team.web_image
          : `${team_images_dir}${team.fullName.replace(/ /g, "_")}.png`;
        const dot_color = team ? hexstring2number(team.color) : 0x000000;

        const dot = new THREE.Mesh(
          geometry,
          new THREE.MeshStandardMaterial({
            color: dot_color,
            roughness: 0.5,
            metalness: 0.5,
          }),
        );
        dot.position.set(player.emb_0, player.emb_1, player.emb_2);
        scene.add(dot);

        let top_n = 5;
        let closestPlayers = players
          .map((player_alt) => {
            let v: Vector3 = [player.emb_0, player.emb_1, player.emb_2];
            let w: Vector3 = [
              player_alt.emb_0,
              player_alt.emb_1,
              player_alt.emb_2,
            ];
            let similarity =
              (v[0] * w[0] + v[1] * w[1] + v[2] * w[2]) /
              (Math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) *
                Math.sqrt(w[0] ** 2 + w[1] ** 2 + w[2] ** 2));
            return { ...player_alt, similarity };
          })
          .sort((a, b) => b.similarity - a.similarity)
          .slice(0, top_n + 1);

        closestPlayers = closestPlayers.filter(
          (player_alt) => player_alt.id !== player.id,
        );

        return {
          id: player.id,
          title: player.name,
          topic: player,
          content: displayPlayerDetails(team_image, player, player_image, closestPlayers, setQuestion),
          team: player.team,
          color: dot_color,
          position: dot.position.clone(),
          mesh: dot,
          explored: false,
        };
      }),
    ).then((newDots: any) => {
      setDots(newDots);
    });

    const highlightGeometry = new THREE.SphereGeometry(0.1, 32, 32);
    const highlightMaterial = new THREE.MeshBasicMaterial({
      color: 0x86a4ff,
      transparent: true,
      opacity: 0.5,
    });
    const highlightMesh = new THREE.Mesh(highlightGeometry, highlightMaterial);
    highlightMesh.visible = false;
    scene.add(highlightMesh);
    highlightMeshRef.current = highlightMesh;

    const axesHelper = new THREE.AxesHelper(5); // Tamaño de los ejes
    scene.add(axesHelper);

    const controls = new OrbitControls(camera, renderer.domElement);
    controlsRef.current = controls;
    camera.position.z = 20;

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener("resize", handleResize);

    const handleClick = (event: MouseEvent) => {
      mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
      mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(scene.children);
      if (intersects.length > 0) {
        const clickedDot = dotsRef.current.find((dot) =>
          dot.position.equals(intersects[0].object.position),
        );
        if (clickedDot) {
          highlightDot(clickedDot);
          setSelectedDotData(clickedDot);
          (clickedDot.mesh.material as THREE.MeshStandardMaterial).color.set(
            0x000000,
          );
          clickedDot.explored = true;
          setIsSearchMode(false); // Set to false when a dot is clicked directly
          setIsLearnMode(false); // Reset learn mode when a new dot is clicked
        }
      }
    };
    window.addEventListener("click", handleClick);

    return () => {
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("click", handleClick);
      mountRef.current?.removeChild(renderer.domElement);
    };
  }, [players]);

  useEffect(() => {
    dotsRef.current.forEach((dot) => {
      const team = teamsData.find((team) => team.fullName === dot.team);
      const dot_color = toggleColor
        ? dot.topic.position == "F"
          ? "#FF0000"
          : dot.topic.position == "M"
            ? "#00FF00"
            : dot.topic.position == "D"
              ? "#0000FF"
              : "#000000"
        : team
          ? hexstring2number(team.color)
          : 0x000000;
      (dot.mesh.material as THREE.MeshStandardMaterial).color.set(dot_color);
    });
  }, [toggleColor]);

  const highlightDot = (dot: DotData) => {
    if (highlightMeshRef.current) {
      highlightMeshRef.current.material = new THREE.MeshBasicMaterial({
        color: dot.color,
        transparent: true,
        opacity: 0.5,
      });
      highlightMeshRef.current.position.copy(dot.position);
      highlightMeshRef.current.visible = true;
    }
  };

  const highlightRelatedDots = (baseId: number, title: string) => {
    // const highlightMaterial = new THREE.MeshBasicMaterial({ color: 0x86A4FF, transparent: true, opacity: 0.5 });

    dots.forEach((dot) => {
      if (Math.floor(dot.id) === baseId || dot.title === title) {
        // Set the highlight material to all matching dots with the same title
        // dot.mesh.material = highlightMaterial.clone();  // Directly assign the new material
        dot.mesh.material = new THREE.MeshStandardMaterial({
          color: dot.color,
          transparent: true,
          opacity: 0.5,
        });
      }
    });
  };

  const resetDotColors = () => {
    // const defaultMaterial = new THREE.MeshStandardMaterial({
    // color: 0x000000
    // color: dot.color
    // });
    dots.forEach((dot) => {
      // dot.mesh.material = defaultMaterial.clone();
      dot.mesh.material = new THREE.MeshStandardMaterial({ color: dot.color });
    });
  };

  const handleNextOrPrevDot = (direction: "next" | "prev") => {
    if (!selectedDotData) return;

    const currentIndex = dots.findIndex((dot) => dot.id === selectedDotData.id);
    const targetDot =
      direction === "next"
        ? dots
            .slice(currentIndex + 1)
            .find((dot) => dot.title === selectedDotData.title)
        : dots
            .slice(0, currentIndex)
            .reverse()
            .find((dot) => dot.title === selectedDotData.title);

    if (targetDot) {
      setSelectedDotData(targetDot);
      highlightDot(targetDot);
      highlightRelatedDots(Math.floor(targetDot.id), targetDot.title);

      if (cameraRef.current && controlsRef.current) {
        const targetPosition = targetDot.position.clone();
        const offset = new THREE.Vector3(0.2, 0.2, 0.5);
        const cameraTargetPosition = targetPosition.clone().add(offset);

        const startPosition = cameraRef.current.position.clone();
        const startControlsTarget = controlsRef.current.target.clone();
        const animationDuration = 1000;
        const startTime = Date.now();

        const animateCamera = () => {
          const elapsedTime = Date.now() - startTime;
          const progress = Math.min(elapsedTime / animationDuration, 1);

          cameraRef.current!.position.lerpVectors(
            startPosition,
            cameraTargetPosition,
            progress,
          );
          controlsRef.current!.target.lerpVectors(
            startControlsTarget,
            targetPosition,
            progress,
          );
          controlsRef.current!.update();

          if (progress < 1) {
            requestAnimationFrame(animateCamera);
          }
        };

        animateCamera();
      }
    }
  };

  const hasNextDot = selectedDotData
    ? dots
        .slice(dots.findIndex((dot) => dot.id === selectedDotData.id) + 1)
        .some((dot) => dot.title === selectedDotData.title)
    : false;

  const hasPrevDot = selectedDotData
    ? dots
        .slice(
          0,
          dots.findIndex((dot) => dot.id === selectedDotData.id),
        )
        .some((dot) => dot.title === selectedDotData.title)
    : false;

  const getTextFromContent = (content: React.ReactElement): string => {
    if (typeof content === "string") {
      return content;
    }
    if (React.isValidElement(content)) {
      const children = (content.props as React.PropsWithChildren<{}>).children;
      if (typeof children === "string") {
        return children;
      }
    }
    return "";
  };

  const handleQuestionSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const query = question.toLowerCase();

    const matchingDots = dots.filter(
      (dot) =>
        dot?.title?.toLowerCase().includes(query) ||
        getTextFromContent(dot.content).toLowerCase().includes(query),
    );

    if (
      matchingDots.length > 0 &&
      sceneRef.current &&
      cameraRef.current &&
      controlsRef.current
    ) {
      const firstDot = matchingDots[0];
      if (!firstDot) return;
      const baseId = Math.floor(firstDot.id);

      const titleMatch = dots?.find(
        (dot) =>
          dot?.title?.toLowerCase().includes(query) && Number.isInteger(dot.id),
      );
      const targetDot = titleMatch || firstDot;

      setSelectedDotData(targetDot);
      highlightDot(targetDot);
      highlightRelatedDots(baseId, targetDot.title); // Highlight all dots with the same base ID or title

      const targetPosition = targetDot.position.clone();
      const offset = new THREE.Vector3(0.2, 0.2, 0.5);
      const cameraTargetPosition = targetPosition.clone().add(offset);

      const startPosition = cameraRef.current.position.clone();
      const startControlsTarget = controlsRef.current.target.clone();
      const animationDuration = 1000;
      const startTime = Date.now();

      const animateCamera = () => {
        const elapsedTime = Date.now() - startTime;
        const progress = Math.min(elapsedTime / animationDuration, 1);

        cameraRef.current!.position.lerpVectors(
          startPosition,
          cameraTargetPosition,
          progress,
        );
        controlsRef.current!.target.lerpVectors(
          startControlsTarget,
          targetPosition,
          progress,
        );
        controlsRef.current!.update();

        if (progress < 1) {
          requestAnimationFrame(animateCamera);
        }
      };

      animateCamera();
      setIsSearchMode(true); // Set to true when a dot is found by searching
      setIsLearnMode(false); // Reset learn mode when a new dot is found by searching
    }

    setQuestion("");
  };

  const handleLearnClick = () => {
    resetDotColors();
    highlightRelatedDots(
      Math.floor(selectedDotData!.id),
      selectedDotData!.title,
    );
    setIsSearchMode(true);
    setIsLearnMode(true);

    // Navigate to the selected dot
    if (cameraRef.current && controlsRef.current) {
      const targetPosition = selectedDotData!.position.clone();
      const offset = new THREE.Vector3(0.2, 0.2, 0.5);
      const cameraTargetPosition = targetPosition.clone().add(offset);

      const startPosition = cameraRef.current.position.clone();
      const startControlsTarget = controlsRef.current.target.clone();
      const animationDuration = 1000;
      const startTime = Date.now();

      const animateCamera = () => {
        const elapsedTime = Date.now() - startTime;
        const progress = Math.min(elapsedTime / animationDuration, 1);

        cameraRef.current!.position.lerpVectors(
          startPosition,
          cameraTargetPosition,
          progress,
        );
        controlsRef.current!.target.lerpVectors(
          startControlsTarget,
          targetPosition,
          progress,
        );
        controlsRef.current!.update();

        if (progress < 1) {
          requestAnimationFrame(animateCamera);
        }
      };

      animateCamera();
    }
  };

  const getCurrentDotIndex = () => {
    if (!selectedDotData) return 0;
    const sameTitleDots = dots.filter(
      (dot) => dot.team === selectedDotData.team,
    );
    return sameTitleDots.findIndex((dot) => dot.id === selectedDotData.id) + 1;
  };

  const getTotalDotsWithTitle = () => {
    if (!selectedDotData) return 0;
    return dots.filter((dot) => dot.team === selectedDotData.team).length;
  };

  return (
    <div className="relative h-screen w-screen bg-gray-100 text-gray-900">
      {/* Toggle Color Mode */}
      <div className="absolute right-4 top-4 flex gap-2">
        <Button
          onClick={() => setToggleColor(!toggleColor)}
          className="bg-black text-white hover:bg-zinc-900"
        >
          {toggleColor ? "Team Color" : "Position Color"}
        </Button>

      {/* Dropdown Select Data */}
        <select
          onChange={(e) => {
            const selectedData = sampleDataMap[e.target.value as keyof typeof sampleDataMap];
            setData(selectedData as unknown as Player[]);
          }}
          className="p-2 bg-white border border-gray-300 rounded-md"
        >
          {Object.keys(sampleDataMap).map((key) => (
            <option key={key} value={key}>
              {key}
            </option>
          ))}
        </select>
      </div>

      <div className="absolute left-4 top-4 w-80 space-y-4">
        <Card className="rounded-lg border border-gray-200 bg-white shadow-lg">
          <CardHeader>
            <CardTitle>
              {selectedDotData ? selectedDotData.title : "Player2Vec Embeddings"}
              {selectedDotData && (
                <span className="ml-2 text-sm text-gray-500">
                  {getCurrentDotIndex()}/{getTotalDotsWithTitle()}
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="mb-4 text-sm">
              {selectedDotData
                ? selectedDotData.content
                : "Clickea en un punto o busca un jugador"}
            </p>
            {selectedDotData && (
              <div className="flex justify-end space-x-2">
                {!isLearnMode && !isSearchMode && (
                  <Button
                    onClick={handleLearnClick}
                    className="bg-black text-white hover:bg-zinc-900"
                  >
                    Zoom
                  </Button>
                )}
                {(isSearchMode || isLearnMode) && (
                  <>
                    <Button
                      onClick={() => handleNextOrPrevDot("prev")}
                      disabled={!hasPrevDot}
                      className="bg-black text-white hover:bg-zinc-900 disabled:bg-gray-400"
                    >
                      <ChevronLeft className="h-4 w-4" />
                    </Button>
                    <Button
                      onClick={() => handleNextOrPrevDot("next")}
                      disabled={!hasNextDot}
                      className="bg-black text-white hover:bg-zinc-900 disabled:bg-gray-400"
                    >
                      <ChevronRight className="h-4 w-4" />
                    </Button>
                  </>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <div ref={mountRef} className="h-full w-full" />
      <div className="absolute bottom-4 left-1/2 w-2/3 max-w-2xl -translate-x-1/2 transform">
        <form onSubmit={handleQuestionSubmit} className="flex gap-2">
          <Input
            type="text"
            placeholder="Buscar jugador... (ej. 'Agüero')"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            className="h-12 flex-grow rounded-full border-gray-300 bg-white px-4 text-gray-900"
          />
          <Button
            type="submit"
            className="h-12 w-12 rounded-full bg-black text-white hover:bg-zinc-900"
          >
            <Send className="fill-white" size={36} />
          </Button>
        </form>
      </div>
    </div>
  );
};

export default Dashboard;

function displayPlayerDetails(team_image: string, player: Player, player_image: string, closestPlayers: { similarity: number; id: string; name: string; position: string; team: string; emb_0: number; emb_1: number; emb_2: number; }[], setQuestion: React.Dispatch<React.SetStateAction<string>>): React.JSX.Element {
  return <>
    <div className="flex flex-col gap-4">
      <span className="flex flex-row items-center gap-2">
        <img src={team_image} alt={player.team} />
        <b>{player.team}</b>- {player.position}
      </span>
      <div 
        className="rounded-lg"
        style={{
          // backgroundColor: "#" + team2color(player.team).toString(16),
          // overflow: 'hidden',
          // textOverflow: 'ellipsis',
          // whiteSpace: 'nowrap',
        }}
      >
      <img
        className="rounded-lg"
        style={{
          WebkitFilter: 'drop-shadow(5px -5px 12px rgba(0, 0, 0, 0.5))',
          filter: 'drop-shadow(5px -5px 12px rgba(0, 0, 0, 0.5))',
        }}
        src={player_image}
        alt={player.name || "Not Found"}
        onError={(e) => (e.currentTarget.src = "/player_images/default.png")} />
      </div>
      <div className="mt-2 flex flex-col gap-2">
        <h3>Closest Players</h3>
        <ul>
          {closestPlayers.map((closest_player, i) => (
            <li
              key={i}
              onClick={() => {
                const query = closest_player.name?.toLowerCase();
                if (!query) return;
                setQuestion(query);
              } }
            >
              <b>{closest_player.name}</b> - {closest_player.team} -{" "}
              {closest_player.position}
            </li>
          ))}
        </ul>
      </div>
    </div>
  </>;
}
